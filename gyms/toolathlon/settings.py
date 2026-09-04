"""Shared runtime settings for Toolathlon benchmark evaluation."""

SUBAGENT_CONTEXT_TOKENS = 256_000
SUBAGENT_RECURSION_LIMIT = 410
DECOMPOSER_RECURSION_LIMIT = 410
DEEPSEEK_REASONING_EFFORT = "high"

# Long-lived benchmark services exposed on the Toolathlon host.  Keep this
# shared between the batch-level fail-fast preflight and the authoritative
# in-container reachability check used by each episode.
MCP_EXTERNAL_TCP_DEPENDENCIES = {
    "canvas": (
        ("Canvas backend", "127.0.0.1", 10001),
        ("Canvas proxy", "127.0.0.1", 20001),
    ),
    "woocommerce": (("WooCommerce", "127.0.0.1", 10003),),
    "emails": (
        ("email IMAP", "127.0.0.1", 1143),
        ("email SMTP", "127.0.0.1", 1587),
    ),
}
