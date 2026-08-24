import { errorFromResponse, MyVistaError, UnsupportedError } from "./errors.ts";

const DEFAULT_BASE_URL = "http://127.0.0.1:47317";
const RETRY_STATUSES = new Set([408, 409, 429, 500, 502, 503, 504]);

const QUALITY_POLICY: Record<string, string> = {
  high: "quality_first",
  standard: "balanced",
  low: "cost_first",
  fast: "latency_first",
};

export type MyVistaOptions = {
  apiKey?: string;
  baseUrl?: string;
  timeoutMs?: number;
  maxRetries?: number;
  fetch?: typeof fetch;
};

export type FabricProvenance = {
  requestedModel?: string;
  servedModel?: string;
  provider?: string;
  policy?: string;
  failovers?: number;
  requestId?: string;
};

export type ChatMessage = { role: string; content: string };

export type ChatCompletion = {
  id: string;
  model: string;
  text: string;
  requestId?: string;
  fabric: FabricProvenance;
  raw: Record<string, unknown>;
};

export type ChatChunk = {
  id: string;
  model: string;
  delta: string;
  finishReason?: string | null;
  raw: Record<string, unknown>;
};

function resolveBaseUrl(baseUrl?: string): string {
  let raw = (baseUrl ?? process.env.MYVISTA_BASE_URL ?? DEFAULT_BASE_URL).replace(/\/$/, "");
  if (raw.endsWith("/v1")) raw = raw.slice(0, -3);
  return raw;
}

function provenance(headers: Headers): FabricProvenance {
  const failovers = headers.get("x-fabric-failovers");
  return {
    requestedModel: headers.get("x-fabric-requested-model") ?? undefined,
    servedModel: headers.get("x-fabric-served-model") ?? undefined,
    provider: headers.get("x-fabric-provider") ?? undefined,
    policy: headers.get("x-fabric-policy") ?? undefined,
    failovers: failovers ? Number(failovers) : undefined,
    requestId: headers.get("x-fabric-request-id") ?? undefined,
  };
}

function completionFrom(raw: Record<string, unknown>, headers: Headers): ChatCompletion {
  const choices = (raw.choices as Array<{ message?: { content?: string } }>) ?? [];
  return {
    id: String(raw.id ?? ""),
    model: String(raw.model ?? ""),
    text: String(choices[0]?.message?.content ?? ""),
    requestId: headers.get("x-fabric-request-id") ?? undefined,
    fabric: provenance(headers),
    raw,
  };
}

export class MyVista {
  readonly baseUrl: string;
  readonly apiKey: string | undefined;
  readonly timeoutMs: number;
  readonly maxRetries: number;
  private readonly fetchImpl: typeof fetch;

  readonly chat: { completions: { create: typeof MyVista.prototype.createChat } };
  readonly responses: { create: typeof MyVista.prototype.createResponse };
  readonly embeddings: { create: () => Promise<never> };
  readonly intents: { classify: typeof MyVista.prototype.classify };
  readonly routes: { preview: typeof MyVista.prototype.preview };
  readonly evals: { run: typeof MyVista.prototype.runEval };
  readonly traces: { get: typeof MyVista.prototype.getTrace; list: typeof MyVista.prototype.listTraces };
  readonly agents: { run: () => Promise<never> };

  constructor(options: MyVistaOptions = {}) {
    this.baseUrl = resolveBaseUrl(options.baseUrl);
    this.apiKey = options.apiKey ?? process.env.MYVISTA_API_KEY;
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.maxRetries = options.maxRetries ?? 2;
    this.fetchImpl = options.fetch ?? fetch;
    this.chat = { completions: { create: this.createChat.bind(this) } };
    this.responses = { create: this.createResponse.bind(this) };
    this.embeddings = {
      create: async () => {
        throw new UnsupportedError(
          "embeddings are not served by this fabric yet. There is no /v1/embeddings route and no vectors are synthesized.",
        );
      },
    };
    this.intents = { classify: this.classify.bind(this) };
    this.routes = { preview: this.preview.bind(this) };
    this.evals = { run: this.runEval.bind(this) };
    this.traces = { get: this.getTrace.bind(this), list: this.listTraces.bind(this) };
    this.agents = {
      run: async () => {
        throw new UnsupportedError(
          "agents are not served by this fabric yet. There is no agent runtime and no run is synthesized.",
        );
      },
    };
  }

  async createChat(args: {
    model?: string;
    messages: ChatMessage[];
    stream?: boolean;
    temperature?: number;
    max_tokens?: number;
    [key: string]: unknown;
  }): Promise<ChatCompletion | AsyncIterable<ChatChunk>> {
    const { stream = false, ...rest } = args;
    const response = await this.request("POST", "/v1/chat/completions", {
      model: args.model ?? "auto",
      stream,
      ...rest,
    });
    if (stream) return this.iterateSse(response);
    const raw = (await response.json()) as Record<string, unknown>;
    return completionFrom(raw, response.headers);
  }

  async createResponse(args: {
    input: string;
    model?: string;
    quality?: string;
    latency_slo_ms?: number;
  }): Promise<{ id: string; model: string; output_text: string; requestId?: string; fabric: FabricProvenance }> {
    const messages = [{ role: "user", content: args.input }];
    let model = args.model ?? "auto";
    const policy = args.quality ? QUALITY_POLICY[args.quality.toLowerCase()] : undefined;
    if (policy || args.latency_slo_ms != null) {
      const preview = await this.preview({
        model,
        messages,
        policy,
        latency_slo_ms: args.latency_slo_ms,
      });
      const selected = (preview.selected ?? {}) as { id?: string; model_id?: string };
      if (selected.model_id || selected.id) model = selected.model_id ?? selected.id ?? model;
    }
    const completion = (await this.createChat({ model, messages })) as ChatCompletion;
    return {
      id: completion.id,
      model: completion.model,
      output_text: completion.text,
      requestId: completion.requestId,
      fabric: completion.fabric,
    };
  }

  async classify(input: string, language = "en"): Promise<Record<string, unknown>> {
    const response = await this.request("POST", "/v1/intents/classify", { input, language });
    return (await response.json()) as Record<string, unknown>;
  }

  async preview(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    const response = await this.request("POST", "/v1/routes/preview", { model: "auto", ...body });
    return (await response.json()) as Record<string, unknown>;
  }

  async runEval(suite = "ci"): Promise<Record<string, unknown>> {
    const response = await this.request("POST", "/v1/evals/run", { suite });
    return (await response.json()) as Record<string, unknown>;
  }

  async listTraces(): Promise<Record<string, unknown>> {
    const response = await this.request("GET", "/v1/observability/traces");
    return (await response.json()) as Record<string, unknown>;
  }

  async getTrace(traceId: string): Promise<Record<string, unknown>> {
    const response = await this.request("GET", `/v1/observability/traces/${traceId}`);
    return (await response.json()) as Record<string, unknown>;
  }

  private async request(method: string, path: string, body?: unknown): Promise<Response> {
    const requestId = `req_${crypto.randomUUID().replaceAll("-", "")}`;
    const headers: Record<string, string> = {
      accept: "application/json",
      "x-request-id": requestId,
    };
    if (this.apiKey) headers.authorization = `Bearer ${this.apiKey}`;
    if (body !== undefined) headers["content-type"] = "application/json";

    let lastError: unknown;
    for (let attempt = 0; attempt <= this.maxRetries; attempt += 1) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      try {
        const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
          method,
          headers,
          body: body === undefined ? undefined : JSON.stringify(body),
          signal: controller.signal,
        });
        if (response.ok) return response;
        if (attempt >= this.maxRetries || !RETRY_STATUSES.has(response.status)) {
          let parsed: unknown = null;
          try {
            parsed = await response.json();
          } catch {
            parsed = null;
          }
          throw errorFromResponse(
            response.status,
            parsed,
            response.headers.get("x-fabric-request-id"),
          );
        }
        const retryAfter = Number(response.headers.get("retry-after"));
        await delay(Number.isFinite(retryAfter) ? retryAfter * 1000 : 250 * 2 ** attempt);
      } catch (error) {
        lastError = error;
        if (error instanceof MyVistaError) throw error;
        if (attempt >= this.maxRetries) {
          throw new MyVistaError(error instanceof Error ? error.message : "request failed", {
            errorType: "api_connection_error",
          });
        }
        await delay(250 * 2 ** attempt);
      } finally {
        clearTimeout(timer);
      }
    }
    throw lastError instanceof Error ? lastError : new MyVistaError("request failed");
  }

  private async *iterateSse(response: Response): AsyncIterable<ChatChunk> {
    const text = await response.text();
    for (const block of text.split("\n\n")) {
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!data || data === "[DONE]") continue;
      let parsed: Record<string, unknown>;
      try {
        parsed = JSON.parse(data) as Record<string, unknown>;
      } catch {
        continue;
      }
      if (parsed.error && !parsed.choices) {
        throw errorFromResponse(200, parsed, null);
      }
      const choices = (parsed.choices as Array<{ delta?: { content?: string }; finish_reason?: string }>) ?? [];
      yield {
        id: String(parsed.id ?? ""),
        model: String(parsed.model ?? ""),
        delta: String(choices[0]?.delta?.content ?? ""),
        finishReason: choices[0]?.finish_reason ?? null,
        raw: parsed,
      };
    }
  }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
