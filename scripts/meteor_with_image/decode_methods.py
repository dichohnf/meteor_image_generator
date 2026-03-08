import numpy as np
import torch
from PIL import Image

from main import DataModuleFromConfig
from scripts.meteor_with_image.input import Options


def load_image(path) -> torch.Tensor:
    loaded_img = Image.open(path).convert("RGB")
    img_np = np.array(loaded_img)
    # Reverse the normalization: from [0, 255] back to [-1.0, 1.0]
    img_np = (img_np.astype(np.float32) / 127.5) - 1.0
    img_np = img_np.transpose(2, 0, 1)
    restored_tensor = torch.from_numpy(img_np)
    restored_tensor = restored_tensor.to('cuda')
    return restored_tensor

def decode_message(options: Options, model: torch.nn.Module, dsets: DataModuleFromConfig, image:torch.Tensor) -> str:
    pass