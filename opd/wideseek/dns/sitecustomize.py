"""Keep bare-metal worker DNS on the existing socket-based hostname override."""
import asyncio

from opd.synth.mlspace.dns import sitecustomize  # noqa: F401

try:
    import uvloop
except ImportError:
    pass
else:
    # Agent Server imports this factory. uvloop's native resolver bypasses
    # socket.getaddrinfo, so use asyncio for this process and its workers.
    uvloop.new_event_loop = asyncio.SelectorEventLoop
