import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_flexiv_example() -> dict:

    return {
        "observation/state": np.random.rand(9),  # TCP姿态有9个维度（位置和方向）
        "observation/image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),  # 随机生成的RGB图像
        "observation/wrist_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),  # 随机生成的手腕相机RGB图像
        "prompt": "Grasp and place an object.",  # 默认提示文本
    }


def _parse_image(image) -> np.ndarray:

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        # 如果图像是浮点类型（0-1范围），转换为uint8（0-255范围）
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        # 如果图像格式为(C,H,W)，转换为(H,W,C)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FlexivInputs(transforms.DataTransformFn):


    action_dim: int

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:

        mask_padding = self.model_type == _model.ModelType.PI0

        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": np.zeros_like(base_image),  # 主相机图像（不存在，用零填充）
                "left_wrist_0_rgb": wrist_image,  # 左手腕相机图像
       
                "right_wrist_0_rgb": np.zeros_like(base_image),  # 右手腕相机图像（不存在，用零填充）
            },
            "image_mask": {
                "base_0_rgb": np.False_ if mask_padding else np.True_,  # 主相机图像不存在
                "left_wrist_0_rgb": np.True_,  # 左手腕相机图像存在
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,  # 右手腕相机图像不存在
            },
        }

  
        if "actions" in data:
   
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "tasks" in data:  # 如果使用tasks字段存储指令
            inputs["prompt"] = data["tasks"]
            
       
        return inputs


@dataclasses.dataclass(frozen=True)
class FlexivOutputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
 
        return {"actions": np.asarray(data["actions"][:16, :10])} 