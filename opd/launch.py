"""Shared single-node GPU layout for OPD launchers."""


def add_gpu_arguments(parser):
    gpu = parser.add_mutually_exclusive_group(required=True)
    gpu.add_argument('--gpu', dest='gpus', nargs=1, type=int,
                     help='Colocate training and rollout on one GPU')
    gpu.add_argument('--gpus', nargs='+', type=int,
                     help='Training GPU indices first, followed by rollout GPU indices')
    parser.add_argument('--training-gpus', type=int, default=1)


def gpu_overrides(gpus, training_gpus):
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError('Choose distinct nonnegative GPU indices')
    single = len(gpus) == 1
    if not 1 <= training_gpus <= (1 if single else len(gpus) - 1):
        raise ValueError('Reserve at least one training GPU and one rollout GPU, or share one GPU')
    overrides = [f'trainer.n_gpus_per_node={training_gpus}', 'trainer.nnodes=1',
                 f'rollout.n_gpus_per_node={1 if single else len(gpus) - training_gpus}',
                 f'rollout.nnodes={0 if single else 1}',
                 f'async_training.use_dynamic_resource_scheduling={str(single).lower()}']
    if single:
        overrides.append('async_training.staleness_threshold=0.0')
    return overrides
