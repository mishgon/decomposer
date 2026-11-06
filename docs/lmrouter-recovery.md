# Lmrouter Outage Recovery

Every `lmrouter/` profile from `decomposer.models.create_model` uses the same
recovering HTTP transport for synchronous and asynchronous inference. This also
covers calls through `model.root_client` and `model.root_async_client`, including
OPD prompt-logprob scoring. Callers do not need their own model retry loops.

Connection/read/write failures, request timeouts, remote protocol errors, and HTTP
408, 429, 500, 502, 503, and 504 pause the request. Retries continue until recovery
or caller cancellation. The delay uses exponential backoff with jitter, starting
at 2.5–5 seconds and capped at 150–300 seconds. SDK retries are disabled for these
profiles so one layer owns recovery.

The transport buffers each inference response before returning it. A failed
response is discarded and the same JSON request is retried. It never replays
agent tools. Inference may have completed on the server before a connection was
lost, so retries can repeat inference and incur additional usage. Registered
profiles disable streaming; this transport is intended for those buffered calls.

Authentication, permission, missing-model, invalid-request, and certificate
verification errors remain visible. Logs include error types or HTTP status,
retry counts, delays, and recovery, without prompts, keys, or response bodies.
Local vLLM and other providers retain their existing retry behavior.

## Deadlines and Deployment

Recovery does not suppress cancellation or extend caller deadlines. A surrounding
`asyncio.timeout` or `wait_for` can still end the request. In particular, this
transport alone does not freeze an episode's wall-clock budget or make an entire
OPD job outage-proof. Existing episode-level guards still handle interrupted
collection attempts; budget accounting across worker processes is a separate
concern. Synchronous calls made in background threads must be managed by their
caller because Python cannot forcibly cancel a running thread.

This behavior applies to clients created by the shared model registry. External
services or legacy standalone clients do not acquire it automatically. Running
experiment snapshots retain their original code until explicitly relaunched.
