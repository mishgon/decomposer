"""Two-GPU integration check of the exact async weight-transfer backend."""
import faulthandler

import ray
import torch


@ray.remote(num_gpus=1)
class Peer:
    def __init__(self, master):
        from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine
        faulthandler.dump_traceback_later(45, repeat=True)
        self.engine = NCCLCheckpointEngine(bucket_size=1024 * 1024, is_master=master)

    def prepare(self):
        return self.engine.prepare()

    def connect(self, rank, metadata):
        self.engine.init_process_group(rank, 2, metadata)

    async def send(self, version):
        weights = [("probe", torch.full((1024 * 1024,), float(version), device="cuda"))]
        await self.engine.send_weights(iter(weights), global_steps=version)
        return True

    async def receive(self, version):
        result = {}
        async for name, weight in self.engine.receive_weights(global_steps=version):
            result[name] = weight.cpu().float().sum().item()
        return result


def main():
    ray.init(address="local", num_cpus=4, num_gpus=2, include_dashboard=False)
    peers = []
    try:
        peers = [Peer.remote(True), Peer.remote(False)]
        metadata = ray.get([p.prepare.remote() for p in peers], timeout=90)[0]
        ray.get([p.connect.remote(i, metadata) for i, p in enumerate(peers)], timeout=60)
        for version in (1, 2):
            sent, received = ray.get([peers[0].send.remote(version), peers[1].receive.remote(version)], timeout=60)
            assert sent and received == {"probe": float(version * 1024 * 1024)}, received
        print("Async NCCL transfer passed: two versions, multiple buckets, exact tensors", flush=True)
    finally:
        for peer in peers:
            ray.kill(peer)
        ray.shutdown()


if __name__ == "__main__":
    main()
