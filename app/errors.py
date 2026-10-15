"""OpenAI error envelope and the gateway's status/code mapping (docs/api-contract.vi.md §7).

`APIError` is raised by any layer (scheduler, engine adapter, endpoints). HTTP handlers render it as a JSON
envelope; the realtime adapter reuses `type`/`code`/`message`/`param` for its `error` events (their `status` is
unused there). Messages must never contain audio, transcripts, tokens or internal paths.
"""

import json

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Scope, Send

from app.schemas import ErrorBody, ErrorResponse


class APIError(Exception):
    def __init__(
        self,
        status: int,
        type_: str,
        code: str,
        message: str,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(message)
        self.status, self.type, self.code, self.message, self.param = (
            status,
            type_,
            code,
            message,
            param,
        )
        self.headers = headers or {}

    def body(self) -> dict:
        body = ErrorBody(message=self.message, type=self.type, param=self.param, code=self.code)
        return ErrorResponse(error=body).model_dump()

    def response(self) -> JSONResponse:
        return JSONResponse(self.body(), status_code=self.status, headers=self.headers)


def invalid_request(code: str, message: str, param: str | None = None) -> APIError:
    """400: missing_required_parameter, unsupported_parameter, invalid_value, unsupported_language, ..."""
    return APIError(400, "invalid_request_error", code, message, param)


def invalid_api_key() -> APIError:
    return APIError(401, "authentication_error", "invalid_api_key", "Missing or invalid API key.")


def model_not_found(model: str) -> APIError:
    # The id is echoed only if it is short and printable; it is client input, not a secret.
    shown = model if len(model) <= 64 and model.isprintable() else "<invalid>"
    return APIError(
        404,
        "invalid_request_error",
        "model_not_found",
        f"The model '{shown}' does not exist.",
        "model",
    )


def not_found() -> APIError:
    return APIError(404, "invalid_request_error", "not_found", "Not found.")


def request_too_large(limit_bytes: int, what: str = "Request body") -> APIError:
    return APIError(
        413, "invalid_request_error", "request_too_large", f"{what} exceeds {limit_bytes} bytes."
    )


def invalid_audio(message: str = "Audio file could not be decoded.") -> APIError:
    return APIError(400, "invalid_request_error", "invalid_audio", message, "file")


def audio_too_long(limit_seconds: float) -> APIError:
    return APIError(
        400,
        "invalid_request_error",
        "audio_too_long",
        f"Audio is longer than {limit_seconds:g} seconds.",
        "file",
    )


def server_busy(retry_after: int | None = None) -> APIError:
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return APIError(
        429, "rate_limit_error", "server_busy", "Server is busy, retry later.", headers=headers
    )


def inference_failed() -> APIError:
    return APIError(500, "server_error", "inference_failed", "Inference failed.")


def alignment_failed() -> APIError:
    """The tokens did not match the transcript: timestamps cannot be produced (never guessed)."""
    return APIError(500, "server_error", "alignment_failed", "Timestamps could not be computed.")


def internal_error() -> APIError:
    return APIError(500, "server_error", "internal_error", "Internal server error.")


def model_not_ready() -> APIError:
    return APIError(
        503, "server_error", "model_not_ready", "Model is not ready.", headers={"Retry-After": "5"}
    )


def shutting_down() -> APIError:
    return APIError(503, "server_error", "shutting_down", "Server is shutting down.")


def request_timeout() -> APIError:
    return APIError(503, "server_error", "request_timeout", "Request timed out.")


# -- realtime `error` events (docs/api-contract.vi.md §6) ---------------------------------------------------------


def invalid_event(message: str, param: str | None = None) -> APIError:
    return invalid_request("invalid_event", message, param)


def unsupported_event() -> APIError:
    return invalid_request(
        "unsupported_event", "This event type is not supported by this server.", "type"
    )


def unknown_parameter(param: str) -> APIError:
    """`param` is the dotted path of the client's key, cut to 64 characters."""
    return invalid_request(
        "unknown_parameter",
        "Unknown parameter.",
        param[:64] if param.isprintable() else "<invalid>",
    )


def input_audio_buffer_full(limit_seconds: float) -> APIError:
    return invalid_request(
        "input_audio_buffer_full",
        f"Input audio buffer would exceed {limit_seconds:g} seconds; commit or clear it first.",
    )


def input_audio_buffer_commit_empty() -> APIError:
    return invalid_request("input_audio_buffer_commit_empty", "Input audio buffer is empty.")


def too_many_pending_items(limit: int) -> APIError:
    return invalid_request(
        "too_many_pending_items",
        f"More than {limit} committed items are waiting for transcription; commit again later.",
    )


def idle_timeout(seconds: float) -> APIError:
    return invalid_request(
        "idle_timeout", f"No client event for {seconds:g} seconds; closing the session."
    )


def session_expired(seconds: float) -> APIError:
    return invalid_request(
        "session_expired", f"Session reached its maximum duration of {seconds:g} seconds."
    )


# -- rendering ----------------------------------------------------------------------------------------------------


async def send_error(scope: Scope, send: Send, error: APIError) -> None:
    """Raw ASGI error response; for a WebSocket handshake, an HTTP denial response before accept."""
    body = json.dumps(error.body()).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    headers += [(k.lower().encode(), v.encode()) for k, v in error.headers.items()]
    if scope["type"] == "http":
        await send({"type": "http.response.start", "status": error.status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
    elif "websocket.http.response" in (scope.get("extensions") or {}):
        await send(
            {"type": "websocket.http.response.start", "status": error.status, "headers": headers}
        )
        await send({"type": "websocket.http.response.body", "body": body})
    else:  # server without the denial-response extension: refuse the handshake (403)
        await send({"type": "websocket.close", "code": 1008})


def install_handlers(app: FastAPI) -> None:
    """Never let FastAPI/Starlette emit their default `{"detail": ...}` bodies (422 included)."""

    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError) -> JSONResponse:
        return exc.response()

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        loc = errors[0].get("loc", ()) if errors else ()
        param = (
            ".".join(str(p) for p in loc if p not in ("body", "query", "path", "header")) or None
        )
        message = f"Invalid value for '{param}'." if param else "Invalid request."
        return invalid_request("invalid_value", message, param).response()

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 404:
            return not_found().response()
        if exc.status_code == 405:
            return APIError(
                405, "invalid_request_error", "method_not_allowed", "Method not allowed."
            ).response()
        err = APIError(
            exc.status_code,
            "invalid_request_error" if exc.status_code < 500 else "server_error",
            "invalid_request" if exc.status_code < 500 else "internal_error",
            "Request failed.",
        )
        return err.response()
