
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from itertools import chain
import os
from pathlib import Path
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
import time

from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature






def load_policy():
    


    pretrained_path = Path("outputs/train/pickplace_baseline/checkpoints/last/pretrained_model")

    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        # "observation.images.sceneDepth": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        # "observation.force": PolicyFeature(type=FeatureType.FORCE, shape=(15,)),
    }

    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }



    # 模拟 config
    config = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        device="cuda",
        pretrained_path = Path("outputs/train/pickplace_baseline/checkpoints/last/pretrained_model")
    )


    # 模型初始化
    instance = SmolVLAPolicy(config)

    # 模型文件路径
    # model_id = "./your_local_model_dir"  # 替换为你保存模型的路径
    model_file = os.path.join(pretrained_path, "model.safetensors")

    # strict 模式选择
    strict = False

    # 加载权重
    policy = SmolVLAPolicy._load_as_safetensor(instance, model_file, config.device, strict)
    policy.to(config.device)
    policy.eval()

    # config = SmolVLAConfig(pretrainedconfig)
    # model = ACT(config)
    for name, param in policy.named_parameters():
        print(name, torch.sum(param).item())
        break  # 打印第一个就行，足够检测差异
    return policy

load_policy()
