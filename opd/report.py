"""Generate the standard PDF after a successful OPD run, without changing its status."""
import logging
import os
from pathlib import Path
import subprocess


def save_report(training_output):
    run = training_output.parent if training_output.name == 'train' else training_output
    script = Path(__file__).resolve().parents[1] / 'scripts/pdf-report.py'
    log = run / 'pdf-report.log'
    try:
        with log.open('w') as output:
            subprocess.run([os.environ.get('UV_BIN', 'uv'), 'run', '--no-project',
                            '--script', str(script), str(run)],
                           stdout=output, stderr=subprocess.STDOUT, check=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as error:
        logging.warning('OPD completed, but PDF generation failed (%s); see %s',
                        type(error).__name__, log)
    else:
        logging.info('OPD report saved: %s', run / 'opd-report.pdf')
