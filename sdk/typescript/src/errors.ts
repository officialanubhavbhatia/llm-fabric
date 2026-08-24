export class MyVistaError extends Error {
  readonly errorType: string;
  readonly statusCode: number | null;
  readonly requestId: string | null;
  readonly retryAfterS: number | null;

  constructor(
    message: string,
    options: {
      errorType?: string;
      statusCode?: number | null;
      requestId?: string | null;
      retryAfterS?: number | null;
    } = {},
  ) {
    super(message);
    this.name = "MyVistaError";
    this.errorType = options.errorType ?? "api_error";
    this.statusCode = options.statusCode ?? null;
    this.requestId = options.requestId ?? null;
    this.retryAfterS = options.retryAfterS ?? null;
  }
}

export class AuthenticationError extends MyVistaError {
  constructor(message: string, options: ConstructorParameters<typeof MyVistaError>[1] = {}) {
    super(message, { ...options, errorType: "authentication_error", statusCode: 401 });
    this.name = "AuthenticationError";
  }
}

export class AuthorizationError extends MyVistaError {
  constructor(message: string, options: ConstructorParameters<typeof MyVistaError>[1] = {}) {
    super(message, { ...options, errorType: "permission_error", statusCode: 403 });
    this.name = "AuthorizationError";
  }
}

export class InvalidRequestError extends MyVistaError {
  constructor(message: string, options: ConstructorParameters<typeof MyVistaError>[1] = {}) {
    super(message, { ...options, errorType: "invalid_request_error", statusCode: 400 });
    this.name = "InvalidRequestError";
  }
}

export class ModelNotFoundError extends InvalidRequestError {
  constructor(message: string, options: ConstructorParameters<typeof MyVistaError>[1] = {}) {
    super(message, { ...options, errorType: "model_not_found" });
    this.name = "ModelNotFoundError";
  }
}

export class NotFoundError extends MyVistaError {
  constructor(message: string, options: ConstructorParameters<typeof MyVistaError>[1] = {}) {
    super(message, { ...options, errorType: "not_found", statusCode: 404 });
    this.name = "NotFoundError";
  }
}

export class QuotaExceededError extends MyVistaError {
  constructor(message: string, options: ConstructorParameters<typeof MyVistaError>[1] = {}) {
    super(message, { ...options, errorType: "quota_exceeded", statusCode: 429 });
    this.name = "QuotaExceededError";
  }
}

export class UnsupportedError extends MyVistaError {
  constructor(message: string) {
    super(message, { errorType: "unsupported", statusCode: null });
    this.name = "UnsupportedError";
  }
}

const BY_TYPE: Record<string, new (message: string, options?: object) => MyVistaError> = {
  authentication_error: AuthenticationError,
  permission_error: AuthorizationError,
  invalid_request_error: InvalidRequestError,
  model_not_found: ModelNotFoundError,
  not_found: NotFoundError,
  quota_exceeded: QuotaExceededError,
};

export function errorFromResponse(status: number, body: unknown, requestId: string | null): MyVistaError {
  const error =
    body && typeof body === "object" && "error" in body
      ? (body as { error?: Record<string, unknown> }).error
      : undefined;
  const message = String(error?.message ?? `HTTP ${status}`);
  const type = String(error?.type ?? "");
  const rid = (typeof error?.request_id === "string" ? error.request_id : requestId) ?? null;
  const Ctor = BY_TYPE[type] ?? MyVistaError;
  return new Ctor(message, { statusCode: status, requestId: rid, errorType: type || "api_error" });
}
