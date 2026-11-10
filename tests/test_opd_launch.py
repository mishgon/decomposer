"""CPU-only checks for resource layout and portable MLSpace payloads."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import shlex
import unittest
from unittest.mock import patch

from opd.launch import add_gpu_arguments, gpu_overrides
from opd.synth.mlspace.experiments import ExperimentConfig
from opd.synth.mlspace.jobs import add_launch_arguments, configured_experiments, payload, resolve_paths
from opd.synth.mlspace.run_job import add_runtime_arguments


class LaunchTests(unittest.TestCase):
    def test_gpu_layouts(self):
        for count, trainers in [(1, 1), (2, 1), (3, 2), (4, 2), (8, 4)]:
            with self.subTest(count=count, trainers=trainers):
                parser = argparse.ArgumentParser()
                add_gpu_arguments(parser)
                args = parser.parse_args(['--gpus', *map(str, range(count)), '--training-gpus', str(trainers)])
                values = dict(v.split('=') for v in gpu_overrides(args.gpus, args.training_gpus))
                self.assertEqual(values['trainer.n_gpus_per_node'], str(trainers))
                self.assertEqual(values['rollout.n_gpus_per_node'], str(1 if count == 1 else count - trainers))
                self.assertEqual(values['rollout.nnodes'], '0' if count == 1 else '1')
                self.assertEqual(values['async_training.use_dynamic_resource_scheduling'], str(count == 1).lower())

    def test_reject_invalid_layout(self):
        for gpus, trainers in [([], 1), ([1, 1], 1), ([-1], 1), ([0], 2), ([0, 1], 0), ([0, 1], 2)]:
            with self.subTest(gpus=gpus, trainers=trainers), self.assertRaises(ValueError):
                gpu_overrides(gpus, trainers)

    def args(self, extra=()):
        parser = argparse.ArgumentParser()
        add_launch_arguments(parser)
        args = parser.parse_args(['--author-name', 'researcher', '--author-telegram', '@researcher',
            '--model', '/shared/model with spaces', '--clearml-config', '/shared/clearml.conf',
            '--lmrouter-key-file', '/shared/lmrouter.key', *extra])
        resolve_paths(args, Path('/shared/repository'))
        return args

    def test_required_identity(self):
        parser = argparse.ArgumentParser()
        add_launch_arguments(parser)
        for missing in ('--author-name', '--author-telegram'):
            argv = ['--author-name', 'researcher', '--author-telegram', 'researcher',
                    '--model', '/shared/model', '--clearml-config', '/shared/config']
            index = argv.index(missing)
            del argv[index:index + 2]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(argv)

    def test_payload_roundtrip_without_reading_credentials(self):
        args = self.args(['--num-gpus', '8', '--training-gpus', '4', '--instance-type', 'allocation-eight-gpu'])
        experiment, = configured_experiments(args, [ExperimentConfig('smoke')])
        with patch.dict('os.environ', {'SOME_SECRET': 'do-not-export'}), patch.object(Path, 'read_text', side_effect=AssertionError('read secret')):
            job = payload(experiment, 'train', Path('/shared/code'), 'region', args)
        self.assertNotIn('do-not-export', json.dumps(job))
        self.assertNotIn('HOME', job['env_variables'])
        self.assertEqual(job['job_desc'], 'synth train smoke #researcher @researcher #rnd #occ')
        self.assertTrue(job['env_variables']['OPD_CHECKPOINT_S3_URI'].endswith('/decomposer-smoke'))
        command = shlex.split(job['script'].split(' && ', 1)[1])
        parser = argparse.ArgumentParser()
        parser.add_argument('--experiment')
        parser.add_argument('--mode')
        add_runtime_arguments(parser)
        runtime = parser.parse_args(command[4:])
        self.assertEqual((runtime.num_gpus, runtime.training_gpus), (8, 4))
        self.assertEqual(runtime.model, args.model)
        self.assertEqual(runtime.output, Path('/shared/decomposer-artifacts/sanity/smoke'))

    def test_unknown_instance_and_in_repo_artifacts_rejected(self):
        args = self.args()
        with self.assertRaises(ValueError):
            payload(ExperimentConfig('smoke', num_gpus=8), 'train', Path('/shared/code'), 'region', args)
        args.artifacts_root = Path('/shared/repository/output')
        with self.assertRaises(ValueError):
            resolve_paths(args, Path('/shared/repository'))


if __name__ == '__main__':
    unittest.main()
