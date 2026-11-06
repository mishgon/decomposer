from pathlib import Path
import subprocess

import pytest

from opd import checkpoint_offload as storage

URI = 's3://bucket/training/artifacts/test/global_step_1'


@pytest.fixture
def checkpoint(tmp_path):
    directory = tmp_path / 'global_step_1'
    (directory / 'actor').mkdir(parents=True)
    (directory / 'actor/model.pt').write_bytes(b'adapter weights')
    (directory / 'data.pt').write_bytes(b'loader state')
    (directory / 'evaluation').mkdir()
    (directory / 'evaluation/score.json').write_text('{"score": 1}')
    return directory


@pytest.fixture
def remote(monkeypatch):
    objects = {}
    calls = []
    def command(*args, **kwargs):
        calls.append(args)
        if args[0] == 'ls':
            prefix = args[-1]
            return '\n'.join(f'2026-10-30 12:00 {len(data)} {key}'
                             for key, data in objects.items() if key.startswith(prefix))
        if args[0] == 'sync':
            root = Path(args[1])
            for path in root.rglob('*'):
                if path.is_file():
                    objects[args[2] + str(path.relative_to(root))] = path.read_bytes()
        elif args[0] == 'get':
            prefix, destination = args[-2:]
            for key, data in objects.items():
                if key.startswith(prefix):
                    path = Path(destination) / key[len(prefix):]
                    if not path.exists():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
        return ''
    monkeypatch.setattr(storage, 's3cmd', command)
    return objects, calls


def test_offload_and_restore_preserve_evaluation(checkpoint, remote):
    original = (checkpoint / 'actor/model.pt').read_bytes()
    record = storage.offload(checkpoint, URI)
    assert record['removed_bytes'] == len(original) + len(b'loader state')
    assert not (checkpoint / 'actor/model.pt').exists()
    assert (checkpoint / 'evaluation/score.json').exists()
    assert (checkpoint / storage.MARKER).exists()
    storage.offload(checkpoint, URI)
    assert sum(c[0] == 'sync' for c in remote[1]) == 1
    storage.restore(checkpoint)
    assert (checkpoint / 'actor/model.pt').read_bytes() == original
    assert not (checkpoint / storage.MARKER).exists()


def test_upload_failure_preserves_every_file(checkpoint, monkeypatch):
    def failed(*args, **kwargs):
        if args[0] == 'ls':
            return ''
        raise subprocess.CalledProcessError(1, ['s3cmd', 'sync'])
    monkeypatch.setattr(storage, 's3cmd', failed)
    with pytest.raises(subprocess.CalledProcessError):
        storage.offload(checkpoint, URI)
    assert (checkpoint / 'actor/model.pt').exists()
    assert (checkpoint / 'data.pt').exists()
    assert not (checkpoint / storage.MARKER).exists()


def test_verification_failure_never_deletes(checkpoint, remote, monkeypatch):
    monkeypatch.setattr(storage, 'remote_sizes', lambda uri: {})
    with pytest.raises(RuntimeError, match='verification failed'):
        storage.offload(checkpoint, URI)
    assert (checkpoint / 'actor/model.pt').exists()
    assert not (checkpoint / storage.MARKER).exists()


def test_interrupted_cleanup_leaves_marker_and_can_retry(checkpoint, remote, monkeypatch):
    real_unlink = Path.unlink
    def interrupted(path, *args, **kwargs):
        if path.name == 'data.pt':
            raise OSError('interrupted deletion')
        return real_unlink(path, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(Path, 'unlink', interrupted)
        with pytest.raises(OSError):
            storage.offload(checkpoint, URI)
    assert (checkpoint / storage.MARKER).exists()
    storage.offload(checkpoint, URI)
    assert not (checkpoint / 'data.pt').exists()


def test_failed_restore_keeps_marker(checkpoint, remote):
    storage.offload(checkpoint, URI)
    del remote[0][URI + '/data.pt']
    with pytest.raises(RuntimeError, match='verification failed'):
        storage.restore(checkpoint)
    assert (checkpoint / storage.MARKER).exists()


def test_changed_local_residue_is_not_deleted(checkpoint, remote):
    storage.offload(checkpoint, URI)
    path = checkpoint / 'data.pt'
    path.write_bytes(b'incomplete restore')
    with pytest.raises(RuntimeError, match='Local checkpoint changed'):
        storage.offload(checkpoint, URI)
    assert path.read_bytes() == b'incomplete restore'


def test_cleanup_cannot_follow_directory_symlink(checkpoint, remote, tmp_path):
    storage.offload(checkpoint, URI)
    outside = tmp_path / 'outside'
    outside.mkdir()
    weight = outside / 'model.pt'
    weight.write_bytes(b'adapter weights')
    (checkpoint / 'actor').rmdir()
    (checkpoint / 'actor').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        storage.offload(checkpoint, URI)
    assert weight.read_bytes() == b'adapter weights'
