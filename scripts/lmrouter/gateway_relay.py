"""Relay encrypted router traffic with a TCP MSS that fits a 1500-byte path."""
import argparse
import asyncio
import socket


async def relay(reader, writer, host):
    upstream = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1460)
        sock.setblocking(False)
        await asyncio.wait_for(
            asyncio.get_running_loop().sock_connect(sock, (host, 443)), timeout=10)
        remote, upstream = await asyncio.open_connection(sock=sock)

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
        print(f"Router connection failed: {type(error).__name__}", flush=True)
    finally:
        writer.close()
        if upstream:
            upstream.close()
        else:
            sock.close()


async def main(args):
    server = await asyncio.start_server(
        lambda reader, writer: relay(reader, writer, args.host),
        "127.0.0.1", 18445)
    print(f"Gateway relay ready: 127.0.0.1:18445 -> {args.host}:443 (MSS 1460)",
          flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    asyncio.run(main(parser.parse_args()))
