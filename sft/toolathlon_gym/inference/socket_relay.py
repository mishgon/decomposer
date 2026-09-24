"""Expose a loopback SSH tunnel to task containers through a private Unix socket.

This relays bytes only. Model clients still verify the upstream TLS hostname.
"""
import argparse
import asyncio
import os
from pathlib import Path


async def relay(reader, writer, port):
    upstream = None
    try:
        remote, upstream = await asyncio.open_connection("127.0.0.1", port)

        async def copy(source, destination):
            while data := await source.read(65536):
                destination.write(data)
                await destination.drain()

        tasks = [asyncio.create_task(copy(reader, upstream)),
                 asyncio.create_task(copy(remote, writer))]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    except OSError as error:
        print(f"Tunnel connection failed: {type(error).__name__}", flush=True)
    finally:
        writer.close()
        if upstream:
            upstream.close()


async def main(args):
    args.socket.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.umask(0o077)
    # Refuse to replace an existing socket: another relay may still own it.
    if args.socket.exists():
        raise FileExistsError(args.socket)
    server = await asyncio.start_unix_server(
        lambda reader, writer: relay(reader, writer, args.port), path=args.socket)
    try:
        print(f"Relay ready: {args.socket} -> 127.0.0.1:{args.port}", flush=True)
        async with server:
            await server.serve_forever()
    finally:
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18443)
    asyncio.run(main(parser.parse_args()))
