import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { MyVista, UnsupportedError, AuthenticationError } from "../src/index.ts";
import { errorFromResponse as fromResponse } from "../src/errors.ts";

describe("error mapping", () => {
  it("maps the gateway envelope to a typed error", () => {
    const error = fromResponse(401, { error: { message: "nope", type: "authentication_error" } }, "req_1");
    assert.equal(error instanceof AuthenticationError, true);
    assert.equal(error.requestId, "req_1");
  });
});

describe("unsupported surfaces", () => {
  it("does not invent embeddings or agents", async () => {
    const client = new MyVista({ fetch: async () => new Response("unused") });
    await assert.rejects(() => client.embeddings.create(), UnsupportedError);
    await assert.rejects(() => client.agents.run(), UnsupportedError);
  });
});

describe("chat", () => {
  it("posts /v1/chat/completions and exposes text plus fabric headers", async () => {
    const fetchImpl: typeof fetch = async (input, init) => {
      assert.equal(String(input), "http://127.0.0.1:47317/v1/chat/completions");
      assert.equal(init?.method, "POST");
      const body = JSON.parse(String(init?.body));
      assert.equal(body.model, "auto");
      return new Response(
        JSON.stringify({
          id: "chatcmpl-1",
          model: "cheap",
          choices: [{ message: { role: "assistant", content: "hello" } }],
        }),
        {
          status: 200,
          headers: {
            "content-type": "application/json",
            "x-fabric-request-id": "req_abc",
            "x-fabric-served-model": "cheap",
            "x-fabric-policy": "cost_first",
          },
        },
      );
    };
    const client = new MyVista({ fetch: fetchImpl });
    const response = (await client.chat.completions.create({
      messages: [{ role: "user", content: "hi" }],
    })) as { text: string; fabric: { servedModel?: string }; requestId?: string };
    assert.equal(response.text, "hello");
    assert.equal(response.fabric.servedModel, "cheap");
    assert.equal(response.requestId, "req_abc");
  });
});
