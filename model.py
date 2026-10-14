"""
SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates
Ref. : Hong, Y., Guo, Y.-C., Yi, R., Chen, Y., Cao, Y.-P., & Ma, L. (2025).
Supermat: Physically Consistent PBR Material Estimation at Interactive Rates.
Proceedings of the IEEE/CVF International Conference on Computer Vision,
pp. 25083–25093.
"""

import copy
import logging

import torch
import torch.nn as nn

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel, forward_timestep_embed
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding
from comfy.utils import UNET_MAP_RESNET

UNET_CONFIG = {
    "image_size": 32,
    "in_channels": 4,
    "out_channels": 4,
    "model_channels": 320,
    "num_res_blocks": [2, 2, 2, 2],
    "channel_mult": [1, 2, 4, 4],
    "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
    "transformer_depth_middle": 1,
    "transformer_depth_output": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    "context_dim": 1024,
    "use_linear_in_transformer": True,
    "adm_in_channels": None,
    "num_heads": -1,
    "num_head_channels": 64,
    "attn_precision": torch.float32,
    "use_spatial_transformer": True,
    "use_temporal_attention": False,
    "use_temporal_resblock": False,
    "legacy": False,
}

HEAD_COUNT = 2
TIMESTEP = 999
LATENT_SCALING_FACTOR = 0.18215


# SD 2.1 v-prediction: with zeroed latents, pred_original_sample == V_PREDICTION_SCALE * v.
_betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2
V_PREDICTION_SCALE = -float((1.0 - torch.cumprod(1.0 - _betas, 0)[TIMESTEP]).sqrt())


class SuperMatUNet(UNetModel):
    """UNetModel whose last up block is replicated into one pass per output head.
    diffusers' last up block spans the final three ComfyUI output blocks. Those are
    swapped for empty placeholders, so UNetModel._forward returns the trunk output
    and the three full-resolution skips still queued for it, concatenated on channels.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        tail = list(self.output_blocks[-3:])
        conv_out = self.out[2]
        self.output_blocks = nn.ModuleList(list(self.output_blocks)[:-3] + [nn.Sequential()] * 3)
        self.out_norm = self.out[:2]
        self.out = nn.Identity()
        self.heads = nn.ModuleList([nn.ModuleList(copy.deepcopy(tail)) for _ in range(HEAD_COUNT)])
        self.out_heads = nn.ModuleList([conv_out, copy.deepcopy(conv_out)])
        self.tail_channels = [tail[0][0].in_layers[2].in_channels - self.model_channels] + [self.model_channels] * 3

    def _forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        h = super()._forward(x, timesteps, context, y, control, transformer_options, **kwargs)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype))
        trunk, *skips = h.split(self.tail_channels, dim=1)
        outputs = []
        for head, out_head in zip(self.heads, self.out_heads):
            h = trunk
            for block, skip in zip(head, skips):
                h = forward_timestep_embed(block, torch.cat([h, skip], dim=1), emb, context, transformer_options)
            outputs.append(out_head(self.out_norm(h)))
        return outputs


# ComfyUI's own diffusers sub-key table, used to address a replicated up block
# where unet_to_diffusers would spell out an output_blocks index.
RESNET_SUBS = {v: k for k, v in UNET_MAP_RESNET.items()}

def head_key(name):
    """Diffusers name inside a replicated up block -> ComfyUI name inside SuperMatUNet.heads."""
    kind, index, sub = name.split(".", 2)
    if kind == "resnets":
        return "{}.0.{}".format(index, RESNET_SUBS[sub])
    return "{}.1.{}".format(index, sub)


def to_comfy_state_dict(sd):
    """Rename the diffusers SuperMat checkpoint onto the ComfyUI module layout."""
    key_map = comfy.utils.unet_to_diffusers(UNET_CONFIG)
    comfy_sd = {}
    for key, value in sd.items():
        if key.startswith("last_up_blocks."):
            head, tail = key.split(".", 2)[1:]
            key = "heads.{}.{}".format(head, head_key(tail))
        elif key.startswith("rep_conv_out."):
            head, tail = key.split(".", 2)[1:]
            key = "out_heads.{}.{}".format(head, tail)
        elif key.startswith("conv_norm_out."):
            key = "out_norm." + key_map[key].split(".", 1)[1]
        elif key in key_map:
            key = key_map[key]
        else:
            continue
        comfy_sd[key] = value
    return comfy_sd


class SuperMatModel:
    def __init__(self, model_path):
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        sd, _ = comfy.utils.load_torch_file(model_path, return_metadata=True)
        state_dict = to_comfy_state_dict(sd)
        del sd

        params = sum(v.numel() for v in state_dict.values())
        self.dtype = comfy.model_management.unet_dtype(model_params=params, weight_dtype=torch.float32)
        self.model = SuperMatUNet(dtype=self.dtype, operations=comfy.ops.disable_weight_init, **UNET_CONFIG)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        del state_dict
        if len(missing) > 0 or len(unexpected) > 0:
            raise RuntimeError("{} is not a Base SuperMat checkpoint: missing {}, unexpected {}".format(
                model_path, sorted(missing), sorted(unexpected)))

        self.patcher = comfy.model_patcher.ModelPatcher(self.model, load_device=load_device, offload_device=offload_device)
        if not comfy.model_management.is_device_cpu(offload_device):
            self.model.to(offload_device)
        logging.info("SuperMat load device: %s, offload device: %s, dtype: %s", load_device, offload_device, self.dtype)

    def predict(self, vae, image, context):
        """Run the two heads on [0,1] images; returns raw head latents (one
        [B,4,h,w] tensor per output) for decode_heads. `context` may be None:
        a zero context then replaces the empty-prompt CLIP context, so the
        node runs without a text encoder (tiny output shift)."""
        latent = self.encode(vae, image)
        comfy.model_management.load_models_gpu([self.patcher])
        if context is None:
            context = torch.zeros(1, 77, 1024)
        context = context.to(device=self.patcher.load_device, dtype=self.dtype).expand(latent.shape[0], -1, -1)
        timesteps = torch.full((1,), TIMESTEP, device=self.patcher.load_device, dtype=torch.int64)
        return self.model(latent, timesteps, context=context)

    def decode_heads(self, vae, head_outs):
        """Raw head latents -> decoded [0,1] images (v-prediction rescale + VAE)."""
        return [vae.decode(head_out * V_PREDICTION_SCALE / LATENT_SCALING_FACTOR) for head_out in head_outs]


    def encode(self, vae, image):
        """Encode to the latent posterior mode. VAE.encode() samples it instead, which would make the node non-deterministic."""
        if vae.latent_channels != 4:
            raise ValueError("SuperMat needs the Stable Diffusion VAE, but the given VAE has {} latent channels".format(vae.latent_channels))
        comfy.model_management.load_models_gpu([vae.patcher])
        z = vae.process_input(image.movedim(-1, 1)).to(vae.vae_dtype).to(vae.device)
        z = vae.first_stage_model.quant_conv(vae.first_stage_model.encoder(z))
        return z.chunk(2, dim=1)[0].to(device=self.patcher.load_device, dtype=self.dtype) * LATENT_SCALING_FACTOR
