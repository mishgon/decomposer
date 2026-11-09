"""Opt-in DNS bootstrap for collection hosts without lmrouter DNS resolution.

Prepend this directory to PYTHONPATH before starting collectors and workers.
"""
import asyncio
import socket

_getaddrinfo = socket.getaddrinfo


def _resolve(host, port, *args, **kwargs):
    if host in ('clearml.2a2i.org', 'lmrouter.2a2i.org',
                b'clearml.2a2i.org', b'lmrouter.2a2i.org'):
        host = '176.108.242.226'
    return _getaddrinfo(host, port, *args, **kwargs)


socket.getaddrinfo = _resolve

try:
    import uvloop
except ImportError:
    pass
else:
    # Native uvloop DNS bypasses socket.getaddrinfo.
    uvloop.new_event_loop = asyncio.SelectorEventLoop
