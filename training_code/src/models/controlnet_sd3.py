# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.controlnet import BaseOutput, zero_module
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from torch.nn import functional as F

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 "latent images" for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


@dataclass
class SD3ControlNetOutput(BaseOutput):
    """
    The output of [`SD3ControlNetModel`].

    Args:
        controlnet_block_samples (`tuple[torch.FloatTensor]`):
            A tuple of tensors that if passed to the unet, will (similar to a residual connection) be added to the
            down and up sample inputs. Each sample has a shape of `(batch_size, channel * resolution, height //
            resolution, width // resolution)`.
    """

    controlnet_block_samples: Tuple[torch.FloatTensor]


class SD3ControlNetModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    """
    A ControlNet model for SD3.

    Args:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        patch_size (`int`, *optional*, defaults to 2):
            The patch size to use for the patch embedding.
        in_channels (`int`, *optional*, defaults to 16):
            The number of channels in the input sample.
        num_layers (`int`, *optional*, defaults to 18):
            The number of layers of MMDiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64):
            The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18):
            The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*, defaults to 4096):
            The number of `encoder_hidden_states` dimensions to use.
        caption_projection_dim (`int`, *optional*, defaults to 1152):
            Number of dimensions to use when projecting the `encoder_hidden_states`.
        pooled_projection_dim (`int`, *optional*, defaults to 2048):
            Number of dimensions to use when projecting the `pooled_projections`.
        out_channels (`int`, *optional*, defaults to 16):
            The number of channels in the output sample.
        pos_embed_max_size (`int`, *optional*, defaults to 96):
            The maximum length of the positional embedding.
        conditioning_embedding_out_channels (`tuple`, *optional*, defaults to `(16, 32, 96, 256)`):
            The tuple of output channel for each block in the `conditioning_embedding` layer.
        conditioning_channels (`int`, *optional*, defaults to 3):
            Number of conditioning channels.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        conditioning_embedding_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        conditioning_channels: int = 3,
    ):
        super().__init__()
        default_pos_embed_max_size = 192

        # Validate inputs.
        if len(conditioning_embedding_out_channels) != 4:
            msg = f"Must provide exactly 4 channels for conditioning_embedding_out_channels. Got: {conditioning_embedding_out_channels}."
            raise ValueError(msg)

        self.out_channels = out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=default_pos_embed_max_size,  # hard-code for now.
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)

        # `attention_head_dim` is doubled to account for the mixing.
        # It needs to crafted when we get the actual checkpoints.
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    context_pre_only=i == num_layers - 1,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-06)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        # ControlNet blocks

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=320,
            block_out_channels=self.config.conditioning_embedding_out_channels,
            conditioning_channels=self.config.conditioning_channels,
        )

        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            controlnet_block = nn.Linear(self.inner_dim, self.inner_dim)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale: float = 1.0,
        timestep: Union[torch.Tensor, float, int] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        pooled_projections: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[SD3ControlNetOutput, Tuple[Tuple[torch.FloatTensor, ...], ...]]:
        """
        The [`SD3ControlNetModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor`):
                The noisy residual passed from the UNet.
            controlnet_cond (`torch.FloatTensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, feature_dim)` to be added to the hidden states.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale of the conditioning. The higher the scale, the more the controlnet affects the generation.
            timestep ( `torch.Tensor` or `float` or `int`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            encoder_hidden_states ( `torch.FloatTensor`, *optional*):
                Encoder hidden states.
            pooled_projections ( `torch.FloatTensor`, *optional*):
                Pooled projections for t5 encoders.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.controlnet.SD3ControlNetOutput`] is returned, otherwise a tuple is
            returned where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale") is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Conditioning
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        controlnet_cond = controlnet_cond.view(controlnet_cond.shape[0], -1, controlnet_cond.shape[-1])
        hidden_states = hidden_states + controlnet_cond

        block_res_samples = ()

        for block, controlnet_block in zip(self.transformer_blocks, self.controlnet_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            block_res_sample = controlnet_block(hidden_states)
            block_res_samples = block_res_samples + (block_res_sample,)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        # Apply conditioning scale
        block_res_samples = tuple(conditioning_scale * sample for sample in block_res_samples)

        if not return_dict:
            return (block_res_samples,)

        return SD3ControlNetOutput(controlnet_block_samples=block_res_samples)

    @classmethod
    def from_transformer(
        cls,
        transformer,
        num_layers: int = None,
        conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
        conditioning_channels: int = 3,
        load_weights_from_transformer: bool = True,
    ):
        config = transformer.config
        config["conditioning_embedding_out_channels"] = conditioning_embedding_out_channels
        config["conditioning_channels"] = conditioning_channels

        if num_layers is not None:
            config["num_layers"] = num_layers

        controlnet = cls.from_config(config)

        if load_weights_from_transformer:
            controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            controlnet.time_text_embed.load_state_dict(transformer.time_text_embed.state_dict())
            controlnet.context_embedder.load_state_dict(transformer.context_embedder.state_dict())
            controlnet.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict())
            controlnet.norm_out.load_state_dict(transformer.norm_out.state_dict())
            controlnet.proj_out.load_state_dict(transformer.proj_out.state_dict())

        return controlnet