import torch
import torch.nn as nn

from yolo.utils.model_utils import apply_transfer_freeze


class ToyModel(nn.Module):
    def __init__(self, layers: int = 5):
        super().__init__()
        self.model = nn.ModuleList([nn.Linear(8, 8) for _ in range(layers)])


def test_unfreeze_last_n():
    m = ToyModel(layers=5)
    # ensure initially all params are trainable
    assert any(p.requires_grad for p in m.parameters())

    # unfreeze last 2 layers (should freeze all then unfreeze last 2)
    res = apply_transfer_freeze(m, freeze_first=0, unfreeze_last=2, layer_attr="model")
    assert res["unfrozen"] > 0

    modules = list(m.model)
    for i, mod in enumerate(modules):
        for p in mod.parameters():
            if i >= len(modules) - 2:
                assert p.requires_grad
            else:
                assert not p.requires_grad


def test_freeze_first_n():
    m = ToyModel(layers=5)
    # ensure initially all params are trainable
    assert any(p.requires_grad for p in m.parameters())

    # freeze first 2 layers (should keep others trainable)
    res = apply_transfer_freeze(m, freeze_first=2, unfreeze_last=0, layer_attr="model")
    assert res["frozen"] > 0

    modules = list(m.model)
    for i, mod in enumerate(modules):
        for p in mod.parameters():
            if i < 2:
                assert not p.requires_grad
            else:
                assert p.requires_grad
