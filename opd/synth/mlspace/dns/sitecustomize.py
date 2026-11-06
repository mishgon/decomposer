"""Job-local DNS workaround, loaded at Python startup through PYTHONPATH."""
import socket

_getaddrinfo = socket.getaddrinfo


def _resolve(host, port, *args, **kwargs):
    if host in ('clearml.2a2i.org', 'lmrouter.2a2i.org',
                b'clearml.2a2i.org', b'lmrouter.2a2i.org'):
        host = '176.108.242.226'
    return _getaddrinfo(host, port, *args, **kwargs)


socket.getaddrinfo = _resolve
