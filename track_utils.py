import torch
import torch
from torch.onnx import utils as trace_utils
import numpy as np
from typing import List
import PIL

def fig_to_img(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return PIL.Image.open(buf)



def track_model_graph(model) -> dict:
    """Tracking Model Graph Data Meta-information
        Arguments:
            model (Torch-model): The model being tacked
        Returns:
            dict:
    """
    model_metadata = {}
    try:

        for param_tensor in model.state_dict():
            model_metadata[param_tensor] =  model.state_dict()[param_tensor].size()


    except RuntimeError as e:
        print("Unable to track model graph and hyperparams with error: ",e)
        return model_metadata
    return model_metadata
