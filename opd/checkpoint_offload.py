"""Verified S3 offload using the team's s3cmd and marker convention."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4

MARKER = 'OFFLOADED_TO_S3.json'


def s3cmd(*args, input=None):
    command = [os.environ.get('UV_BIN', 'uv'), 'tool', 'run', 's3cmd',
               '--config', os.environ.get('OPD_S3_CONFIG', str(Path.home() / '.s3cfg')), *args]
    return subprocess.run(command, input=input, capture_output=True, text=True,
                          check=True, timeout=900).stdout


def remote_sizes(uri):
    prefix = uri.rstrip('/') + '/'
    sizes = {}
    for line in s3cmd('ls', '--recursive', prefix).splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) == 4 and fields[2].isdigit() and fields[3].startswith(prefix):
            sizes[fields[3][len(prefix):]] = int(fields[2])
    return sizes


def check_access(uri):
    """Test the job's actual write/read/delete permissions on its own unique probe."""
    probe = uri.rstrip('/') + '/.access-probe-' + uuid4().hex
    content = 'OPD checkpoint storage preflight\n'
    s3cmd('put', '-', probe, input=content)
    try:
        if s3cmd('get', probe, '-') != content:
            raise RuntimeError('S3 checkpoint probe readback differs')
    finally:
        s3cmd('del', probe)


def verify(files, sizes):
    missing = [f['name'] for f in files if sizes.get(f['name']) != f['size']]
    if missing:
        raise RuntimeError(f'Checkpoint size verification failed; local files preserved: {missing}')


def payload_files(directory):
    files = []
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory)
        if relative.parts[0] == 'evaluation' or path.name in {MARKER, MARKER + '.tmp'}:
            continue
        if path.is_symlink():
            raise ValueError(f'Checkpoint payload must not contain symlinks: {path}')
        if path.is_file():
            files.append({'name': str(relative), 'size': path.stat().st_size})
    return files


def local_file(directory, name):
    relative = Path(name)
    path = directory / relative
    if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError('Checkpoint manifest path escapes its directory')
    return path


def offload(directory, uri):
    directory = Path(directory)
    uri = uri.rstrip('/')
    marker = directory / MARKER
    if marker.exists():
        record = json.loads(marker.read_text())
        if record['s3_uri'] != uri:
            raise ValueError('Checkpoint is already archived at a different S3 URI')
        files = record['removed_files']
    else:
        files = payload_files(directory)
        if not files:
            raise ValueError('Refusing to archive an empty checkpoint')
        sizes = remote_sizes(uri)
        if any(sizes.get(f['name']) != f['size'] for f in files):
            s3cmd('sync', str(directory) + '/', uri + '/')
        record = {'offloaded_to_s3': True, 's3_uri': uri,
                  'offloaded_at_utc': datetime.now(timezone.utc).isoformat(),
                  'removed_files': files, 'removed_bytes': sum(f['size'] for f in files),
                  'restore_hint': f'python -m opd.checkpoint_offload restore {directory}'}
    verify(files, remote_sizes(uri))
    # Validate all local residues before deleting any; a partial restore stays marked.
    for f in files:
        path = local_file(directory, f['name'])
        if path.is_symlink() or (path.exists() and path.stat().st_size != f['size']):
            raise RuntimeError(f'Local checkpoint changed; preserving files: {path}')
    temporary = directory / (MARKER + '.tmp')
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(marker)
    # Persist the marker before deletion so an interrupted cleanup is still explicit.
    for f in files:
        local_file(directory, f['name']).unlink(missing_ok=True)
    print(f'Checkpoint offloaded and size-verified: {uri}', flush=True)
    return record


def restore(directory):
    directory = Path(directory)
    marker = directory / MARKER
    record = json.loads(marker.read_text())
    for f in record['removed_files']:
        local_file(directory, f['name'])
    s3cmd('get', '--recursive', '--skip-existing', record['s3_uri'].rstrip('/') + '/', str(directory) + '/')
    sizes = {f['name']: f['size'] for f in payload_files(directory)}
    verify(record['removed_files'], sizes)
    marker.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['restore'])
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    restore(args.checkpoint)
