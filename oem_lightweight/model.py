import torch
import numpy as np
import pickle

from config import config
from fasterseg_api.model_seg import Network_Multi_Path_Infer as Network
from sparsemask_api.sparse_mask_eval_mode import SparseMask


def _torch_load_compat(path, map_location="cpu"):
    """Load legacy checkpoints across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as exc:
        # PyTorch >= 2.6 defaults to weights_only=True, which can break
        # older checkpoints that contain non-tensor pickled objects.
        if "Weights only load failed" in str(exc):
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=map_location)
        raise


def fasterseg(arch, weights):
    # load arch
    state = _torch_load_compat(arch, map_location="cpu")
    model = Network([state["alpha_1_0"].detach(),
                     state["alpha_1_1"].detach(),
                     state["alpha_1_2"].detach()],
                    [None, state["beta_1_1"].detach(),
                     state["beta_1_2"].detach()],
                    [state["ratio_1_0"].detach(),
                     state["ratio_1_1"].detach(),
                     state["ratio_1_2"].detach()],
                    num_classes=config.num_classes,
                    layers=16,
                    Fch=12,
                    width_mult_list=[4. / 12, 6. / 12, 8. / 12, 10. / 12, 1., ],
                    stem_head_width=(8. / 12, 8. / 12),
                    ignore_skip=False)
    model.build_structure([2, 1])

    # load weights
    weights_dict = _torch_load_compat(weights, map_location="cpu")
    state = model.state_dict()
    weights_dict = {k: v for k, v in weights_dict.items() if k in state}
    state.update(weights_dict)
    model.load_state_dict(state)

    return dict(model=model, name="FasterSeg")


def sparsemask(mask, weights):
    # load arch
    mask = np.load(mask)
    model = SparseMask(mask,
                       backbone_name="mobilenet_v2",
                       depth=64,
                       in_channels=3,
                       num_classes=config.num_classes)

    # load weight
    weights_dict = _torch_load_compat(weights, map_location="cpu")
    weights_dict = {key.replace("module.", ""): value for key, value in weights_dict['state_dict'].items()}
    model.load_state_dict(weights_dict, strict=False)

    return dict(model=model, name="SparseMask")

