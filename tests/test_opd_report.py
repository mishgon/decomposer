"""Reporting is a post-training artifact, never a reason to fail completed training."""
import logging
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from opd.report import save_report


def test_report_writes_to_run_folder(tmp_path, monkeypatch):
    monkeypatch.setenv('UV_BIN', '/tools/uv')
    with patch('opd.report.subprocess.run') as run:
        save_report(tmp_path / 'train')
    command = run.call_args.args[0]
    assert command[:4] == ['/tools/uv', 'run', '--no-project', '--script']
    assert Path(command[4]).name == 'pdf-report.py'
    assert command[-1] == str(tmp_path)
    assert (tmp_path / 'pdf-report.log').exists()


@pytest.mark.parametrize('error', [FileNotFoundError('uv'), subprocess.CalledProcessError(1, ['uv']),
                                 subprocess.TimeoutExpired(['uv'], 300)])
def test_report_failure_does_not_fail_training(tmp_path, caplog, error):
    with patch('opd.report.subprocess.run', side_effect=error), caplog.at_level(logging.WARNING):
        save_report(tmp_path / 'train')
    assert 'OPD completed, but PDF generation failed' in caplog.text
