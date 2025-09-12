import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer


# Optional but commonly available in Transformers >=4.38
try:
    from transformers import SiglipTextModel
except Exception:  # pragma: no cover - fallback name
    SiglipTextModel = None  # type: ignore


class _AdapterOutput(tuple):
    """Tuple-like output that also carries a `hidden_states` attribute.

    The SDXL pipeline expects:
      - `out[0]` to be a 2D pooled embedding for the final text encoder
      - `out.hidden_states[-2]` to be the sequence features used for cross-attn
    """

    def __new__(cls, pooled: torch.Tensor, hidden_states_tuple: Tuple[torch.Tensor, ...]):
        obj = tuple.__new__(cls, (pooled,))
        obj.hidden_states = hidden_states_tuple
        return obj


class SiglipAsTextEncoder(nn.Module):
    """Wrap a SigLIP text model to mimic SDXL's `text_encoder_2` contract.

    - Projects SigLIP hidden states to the original CLIP-2 hidden size with `hidden_proj`.
    - Projects pooled features to the original projection dim with `pool_proj`.
    - Exposes a minimal `.config` containing `projection_dim`.
    """

    def __init__(self, siglip_model_id: str, target_hidden_size: int, target_proj_dim: int):
        super().__init__()
        if SiglipTextModel is None:
            raise ImportError("Transformers does not provide SiglipTextModel in this environment.")

        self.model = SiglipTextModel.from_pretrained(siglip_model_id)

        in_hidden = getattr(self.model.config, "hidden_size")
        in_proj = getattr(self.model.config, "projection_dim", in_hidden)

        self.hidden_proj = nn.Linear(in_hidden, target_hidden_size)
        self.pool_proj = nn.Linear(in_proj, target_proj_dim)

        # Minimal config shim used by SDXL to read `projection_dim`
        @dataclass
        class _Config:
            projection_dim: int

        self.config = _Config(projection_dim=target_proj_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, output_hidden_states: bool = True):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Penultimate hidden states if available, else last
        if out.hidden_states and len(out.hidden_states) >= 2:
            penultimate = out.hidden_states[-2]
        else:
            penultimate = out.last_hidden_state

        seq_feats = self.hidden_proj(penultimate)

        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        pooled = self.pool_proj(pooled)

        # Construct a tuple where [-2] works reliably
        hs: Tuple[torch.Tensor, ...] = (torch.empty(0, device=seq_feats.device), seq_feats)
        return _AdapterOutput(pooled, hs)


def swap_text_encoder_2_for_siglip(
    pipe: StableDiffusionXLPipeline,
    siglip_id: str,
    freeze_siglip: bool = True,
    single_encoder: bool = False,
) -> StableDiffusionXLPipeline:
    """Replace `tokenizer_2`/`text_encoder_2` with SigLIP + linear projections.

    The adapter preserves the concatenated text-embedding shape and the pooled
    projection dim expected by SDXL's UNet additional conditioning.
    """

    # Record original TE2 dims to keep pooled projection consistent
    orig_te2 = pipe.text_encoder_2
    if orig_te2 is None:
        raise ValueError("This pipeline does not define text_encoder_2.")

    # For single-encoder mode, we must match UNet cross-attn dim directly.
    # Otherwise, we mimic the original TE2 hidden size (and rely on concat).
    if single_encoder:
        target_hidden = int(pipe.unet.config.cross_attention_dim)
    else:
        try:
            target_hidden = int(getattr(orig_te2.config, "hidden_size"))
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Could not infer original text_encoder_2 hidden size.") from e

    target_proj = int(getattr(orig_te2.config, "projection_dim", target_hidden))

    # Create SigLIP tokenizer and match sequence length with the remaining tokenizer
    tokenizer_2 = AutoTokenizer.from_pretrained(siglip_id, use_fast=True)
    # Ensure matching sequence length with the other tokenizer if present
    if pipe.tokenizer is not None and hasattr(pipe.tokenizer, "model_max_length"):
        tokenizer_2.model_max_length = int(pipe.tokenizer.model_max_length)

    adapter = SiglipAsTextEncoder(siglip_id, target_hidden_size=target_hidden, target_proj_dim=target_proj)

    if freeze_siglip:
        for p in adapter.model.parameters():
            p.requires_grad = False

    pipe.tokenizer_2 = tokenizer_2
    pipe.text_encoder_2 = adapter

    # In single-encoder mode, drop the first encoder entirely.
    if single_encoder:
        pipe.tokenizer = None
        pipe.text_encoder = None
    return pipe


def encode_text(pipe: StableDiffusionXLPipeline, prompts, device, guidance_scale: float = 5.0):
    do_cfg = guidance_scale > 1.0
    out = pipe.encode_prompt(
        prompts,
        device=device,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=[""] * len(prompts) if do_cfg else None,
    )
    return out


def to_latents(vae: AutoencoderKL, images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    images = images.to(dtype)
    latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
    return latents


def add_noise(latents: torch.Tensor, noise_scheduler, noise: Optional[torch.Tensor] = None):
    bsz = latents.shape[0]
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long)
    noise = torch.randn_like(latents) if noise is None else noise
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    return noisy_latents, noise, timesteps


def make_wds_loader(urls: str, batch_size: int, num_workers: int = 4, image_key: str = "jpg", text_key: str = "txt"):
    import webdataset as wds
    from torchvision import transforms
    from PIL import Image

    def decode_img(x):
        with Image.open(x) as im:
            im = im.convert("RGB")
            return im

    transform = transforms.Compose(
        [
            transforms.Resize(1024, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(1024),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = (
        wds.WebDataset(urls, handler=wds.ignore_and_continue)
        .decode()
        .to_tuple(image_key, text_key)
        .map_tuple(lambda img: transform(decode_img(img)), lambda txt: txt.decode("utf-8") if isinstance(txt, (bytes, bytearray)) else str(txt))
    )

    loader = (
        dataset
        .shuffle(1000)
        .batched(batch_size, partial=False)
        .with_length(10**9)  # virtually infinite
        .to_tuple("jpg", "txt")  # for type hints only
    )

    return wds.WebLoader(loader, batch_size=None, num_workers=num_workers)


def save_adapter(adapter: SiglipAsTextEncoder, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "hidden_proj": adapter.hidden_proj.state_dict(),
        "pool_proj": adapter.pool_proj.state_dict(),
        "siglip_config": adapter.model.config.to_dict(),
        "projection_dim": adapter.config.projection_dim,
    }, os.path.join(out_dir, "siglip_adapter.pt"))


def train():
    parser = argparse.ArgumentParser(description="SDXL baseline: use SigLIP (+ linear proj) as the sole text encoder or as a drop-in for CLIP-2")
    parser.add_argument("--pretrained_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--siglip_model", type=str, default="google/siglip-so400m-patch14-384")
    parser.add_argument("--output_dir", type=str, default="outputs/siglip-baseline")
    parser.add_argument("--train_urls", type=str, help="WebDataset shard pattern, e.g. /path/shards/{0000..0999}.tar")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"]) 
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision, project_config=project_config)

    device = accelerator.device

    dtype = torch.float16 if accelerator.mixed_precision == "fp16" else (torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32)

    pipe: StableDiffusionXLPipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=dtype,
        variant=None,
        use_safetensors=True,
    ).to(device)

    # Replace CLIP-2 with SigLIP + linear projections
    pipe = swap_text_encoder_2_for_siglip(pipe, args.siglip_model, freeze_siglip=True, single_encoder=True)

    # Train only the projection layers
    params = list(pipe.text_encoder_2.hidden_proj.parameters()) + list(pipe.text_encoder_2.pool_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    # Use the pipeline's scheduler/vae/unet directly
    noise_scheduler = pipe.scheduler
    vae = pipe.vae
    unet = pipe.unet

    # Data
    assert args.train_urls, "--train_urls must point to WebDataset shards"
    loader = make_wds_loader(args.train_urls, batch_size=args.batch_size, num_workers=args.num_workers)

    unet, vae, optimizer = accelerator.prepare(unet, vae, optimizer)

    step = 0
    pipe.text_encoder_2.hidden_proj.train()
    pipe.text_encoder_2.pool_proj.train()
    unet.train()

    for batch in loader:
        if step >= args.max_steps:
            break

        images, texts = batch  # images: [B,3,1024,1024], texts: list[str]

        with accelerator.accumulate(unet):
            images = images.to(device)

            # Latents
            latents = to_latents(vae, images, dtype=unet.dtype)
            noisy_latents, noise, timesteps = add_noise(latents, noise_scheduler)

            # Text conditioning (uses our swapped SigLIP encoder for pooled + seq embeds)
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encode_text(
                pipe, texts, device=device, guidance_scale=args.guidance_scale
            )

            # Pack added time ids used by SDXL
            add_text_embeds = pooled_prompt_embeds
            add_time_ids = pipe._get_add_time_ids(
                (1024, 1024),
                (0, 0),
                (1024, 1024),
                dtype=unet.dtype,
                text_encoder_projection_dim=int(pipe.text_encoder_2.config.projection_dim),
            )
            add_time_ids = add_time_ids.to(device)

            # CFG
            if args.guidance_scale > 1.0:
                noise_pred_uncond = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=negative_prompt_embeds,
                    added_cond_kwargs={"text_embeds": negative_pooled_prompt_embeds, "time_ids": add_time_ids},
                ).sample
                noise_pred_text = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids},
                ).sample
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids},
                ).sample

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.is_main_process and step % 50 == 0:
            accelerator.print(f"step {step}: loss={loss.item():.4f}")

        step += 1

    if accelerator.is_main_process:
        save_adapter(pipe.text_encoder_2, os.path.join(args.output_dir, "adapter"))

    accelerator.print("Training complete.")


if __name__ == "__main__":
    train()
