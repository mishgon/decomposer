"""Run with the training environment and the pinned, patched verl checkout."""

import pickle

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("verl")
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import left_right_2_no_padding


@pytest.mark.parametrize("lengths", [[5736], [17, 17], [17, 11]])
@pytest.mark.parametrize("channels", [None, 4])
def test_position_ids_minibatch(lengths, channels):
    size = max(lengths)
    positions = torch.arange(size).expand(len(lengths), size)
    if channels:
        positions = positions[:, None, :].expand(-1, channels, -1)
    mask = torch.arange(size)[None, :] < torch.tensor(lengths)[:, None]
    data = TensorDict({"input_ids": torch.ones_like(mask, dtype=torch.long),
                       "attention_mask": mask.long(), "response_mask": mask.long(),
                       "position_ids": positions}, batch_size=[len(lengths)])
    data = pickle.loads(pickle.dumps(left_right_2_no_padding(data)))
    tu.maybe_fix_3d_position_ids(data)
    for i, length in enumerate(lengths):
        batch = tu.index_select_tensor_dict(data, [i])
        actual = batch["position_ids"].unbind()[0]
        torch.testing.assert_close(actual, positions[i, ..., :length])
        assert batch["input_ids"].offsets().diff().item() == length
