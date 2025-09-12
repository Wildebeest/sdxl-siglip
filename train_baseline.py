import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

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
from torch.cuda.amp import autocast
import wandb

# Optional: Backblaze B2 SDK for checkpoint uploads
try:
    from b2sdk.v2 import InMemoryAccountInfo, B2Api
except Exception:  # pragma: no cover
    InMemoryAccountInfo = None  # type: ignore
    B2Api = None  # type: ignore
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


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

    def __init__(self, siglip_model_id: str, target_hidden_size: int, target_proj_dim: int, target_seq_len: int = 77):
        super().__init__()
        if SiglipTextModel is None:
            raise ImportError("Transformers does not provide SiglipTextModel in this environment.")

        self.model = SiglipTextModel.from_pretrained(siglip_model_id)
        self.target_seq_len = int(target_seq_len)

        in_hidden = getattr(self.model.config, "hidden_size")
        in_proj = getattr(self.model.config, "projection_dim", in_hidden)

        self.hidden_proj = nn.Linear(in_hidden, target_hidden_size)
        self.hidden_ln = nn.LayerNorm(target_hidden_size)
        self.pool_proj = nn.Linear(in_proj, target_proj_dim)

        # Lightweight learnable resampler: learnable queries attend over token sequence
        self.resampler = TokenResampler(d_model=target_hidden_size, target_len=self.target_seq_len, num_heads=4)
        self.post_ln = nn.LayerNorm(target_hidden_size)
        self.out_scale = 1.0 / math.sqrt(float(target_hidden_size))

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
        seq_feats = self.hidden_ln(seq_feats)
        # Learnable resampling to exactly target_seq_len tokens
        seq_feats = self.resampler(seq_feats)
        seq_feats = self.post_ln(seq_feats) * self.out_scale

        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        pooled = self.pool_proj(pooled)

        # Construct a tuple where [-2] works reliably
        hs: Tuple[torch.Tensor, ...] = (seq_feats, torch.empty(0, device=seq_feats.device))
        return _AdapterOutput(pooled, hs)

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.get_default_dtype()

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")


class TokenResampler(nn.Module):
    """Learnable token resampler using cross-attention from T learnable queries.

    Inputs:  x [B, L, D]
    Outputs: y [B, T, D]  where T = target_len
    """
    def __init__(self, d_model: int, target_len: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.target_len = int(target_len)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.scale = (d_model // num_heads) ** -0.5
        self.queries = nn.Parameter(torch.randn(self.target_len, d_model) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # conservative init to avoid large outputs at start
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.dropout = nn.Dropout(dropout)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        b, l, d = x.shape
        q = self.ln_q(self.queries).unsqueeze(0).expand(b, -1, -1)  # [B, T, D]
        k = self.k_proj(self.ln_kv(x))      # [B, L, D]
        v = self.v_proj(self.ln_kv(x))      # [B, L, D]

        # reshape to multi-head: [B, H, T, Dh], [B, H, L, Dh]
        h = self.num_heads
        dh = d // h
        q = q.view(b, self.target_len, h, dh).transpose(1, 2)  # [B,H,T,Dh]
        k = k.view(b, l, h, dh).transpose(1, 2)                # [B,H,L,Dh]
        v = v.view(b, l, h, dh).transpose(1, 2)                # [B,H,L,Dh]

        attn = (q @ k.transpose(-2, -1)) * self.scale          # [B,H,T,L]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        y = attn @ v                                           # [B,H,T,Dh]
        y = y.transpose(1, 2).contiguous().view(b, self.target_len, d)  # [B,T,D]
        y = self.out_proj(y)
        return y


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

    # Create SigLIP tokenizer (max length set after model init)
    tokenizer_2 = AutoTokenizer.from_pretrained(siglip_id, use_fast=True)

    target_seq_len = int(getattr(pipe.tokenizer, "model_max_length", 77) or 77)
    adapter = SiglipAsTextEncoder(siglip_id, target_hidden_size=target_hidden, target_proj_dim=target_proj, target_seq_len=target_seq_len)
    # Align tokenizer_2 max length with SigLIP's capacity to avoid 77 vs 64 mismatch
    try:
        siglip_max = int(getattr(adapter.model.config, "max_position_embeddings", tokenizer_2.model_max_length))
        tokenizer_2.model_max_length = siglip_max
        # Also cap tokenizer_1 to the smallest max to keep pair lengths aligned
        if pipe.tokenizer is not None and hasattr(pipe.tokenizer, "model_max_length"):
            pipe.tokenizer.model_max_length = int(min(int(pipe.tokenizer.model_max_length), siglip_max))
    except Exception:
        pass

    if freeze_siglip:
        for p in adapter.model.parameters():
            p.requires_grad = False

    # Move adapter to the same device as UNet (pipeline primary device)
    try:
        dev = getattr(pipe.unet, "device", next(pipe.unet.parameters()).device)
        adapter = adapter.to(device=dev)
    except Exception:
        adapter = adapter.to("cuda" if torch.cuda.is_available() else "cpu")

    pipe.tokenizer_2 = tokenizer_2
    pipe.text_encoder_2 = adapter

    # In single-encoder mode, drop the first encoder entirely.
    if single_encoder:
        pipe.tokenizer = None
        pipe.text_encoder = None
    return pipe


def encode_text(pipe: StableDiffusionXLPipeline, prompts, device, guidance_scale: float = 5.0):
    """Encode prompts for training.

    If we've replaced SDXL text encoders with a single SigLIP adapter (tokenizer is None),
    compute embeddings directly using tokenizer_2/text_encoder_2 and return the 4-tuple that
    SDXL training expects. Otherwise, delegate to pipeline.encode_prompt.
    """
    do_cfg = guidance_scale > 1.0

    # SigLIP-only path (tokenizer removed, tokenizer_2 + text_encoder_2 present)
    if getattr(pipe, "tokenizer", None) is None and hasattr(pipe, "tokenizer_2") and hasattr(pipe, "text_encoder_2"):
        tok = pipe.tokenizer_2
        te = pipe.text_encoder_2
        max_len = int(getattr(tok, "model_max_length", getattr(te, "target_seq_len", 77)) or 77)

        enc = tok(list(prompts), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        with autocast(enabled=False):
            out = te(input_ids=input_ids, attention_mask=None, output_hidden_states=True)
        seq = out.hidden_states[-2]  # [B, 77, cross_attention_dim] after adapter
        pooled = out[0]              # [B, proj_dim]

        # Use UNet dtype for encoder_hidden_states to match attention kernels
        dtype = pipe.unet.dtype
        prompt_embeds = seq.to(dtype=dtype, device=device)
        pooled_prompt_embeds = pooled.to(dtype=dtype, device=device)

        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        if do_cfg:
            neg = tok([""] * len(prompts), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
            n_input_ids = neg["input_ids"].to(device)
            with autocast(enabled=False):
                nout = te(input_ids=n_input_ids, attention_mask=None, output_hidden_states=True)
            nseq = nout.hidden_states[-2]
            npooled = nout[0]
            negative_prompt_embeds = nseq.to(dtype=dtype, device=device)
            negative_pooled_prompt_embeds = npooled.to(dtype=dtype, device=device)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # Default SDXL path
    return pipe.encode_prompt(
        prompts,
        device=device,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=[""] * len(prompts) if do_cfg else None,
    )


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


def build_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(1024, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(1024),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def make_wds_loader(urls: str, batch_size: int, num_workers: int = 4, image_key: str = "jpg", text_key: str = "txt"):
    import io
    import webdataset as wds
    from PIL import Image, ImageFile

    # Be tolerant of truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    def decode_img(x):
        # x can be PIL.Image, bytes, file-like, or a path; normalize to PIL RGB
        if isinstance(x, Image.Image):
            im = x
        else:
            buf = io.BytesIO(x) if isinstance(x, (bytes, bytearray, memoryview)) else x
            with Image.open(buf) as im2:
                im = im2.copy()
        return im.convert("RGB")

    transform = build_train_transform()

    dataset = (
        wds.WebDataset(urls, handler=wds.ignore_and_continue)
        .decode("pil")
        .to_tuple(image_key, text_key)
        .map_tuple(lambda img: transform(decode_img(img)), lambda txt: txt.decode("utf-8") if isinstance(txt, (bytes, bytearray)) else str(txt))
    )

    loader = (
        dataset
        .shuffle(1000)
        .batched(batch_size, partial=False)
        .with_length(10**9)  # virtually infinite
    )

    return wds.WebLoader(loader, batch_size=None, num_workers=num_workers)


def tensor_to_pil(x: torch.Tensor) -> "Image.Image":
    x = x.detach().cpu().clamp(-1, 1)
    x = (x * 0.5 + 0.5).clamp(0, 1)
    return to_pil_image(x)


def sample_training_examples(urls: str, n: int, seed: int = 12345, image_key: str = "jpg", text_key: str = "txt") -> List[Dict[str, object]]:
    import io
    import webdataset as wds
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    def decode_img(x):
        if isinstance(x, Image.Image):
            im = x
        else:
            buf = io.BytesIO(x) if isinstance(x, (bytes, bytearray, memoryview)) else x
            with Image.open(buf) as im2:
                im = im2.copy()
        return im.convert("RGB")

    rng = torch.Generator().manual_seed(seed)
    transform = build_train_transform()
    ds = (
        wds.WebDataset(urls, handler=wds.ignore_and_continue)
        .shuffle(1000, rng=rng)
        .decode("pil")
        .to_tuple(image_key, text_key)
        .map_tuple(lambda img: transform(decode_img(img)), lambda txt: txt.decode("utf-8") if isinstance(txt, (bytes, bytearray)) else str(txt))
    )

    out: List[Dict[str, object]] = []
    for img_t, txt in ds:
        out.append({"prompt": txt, "train_img": tensor_to_pil(img_t)})
        if len(out) >= n:
            break
    return out


def save_adapter(adapter: SiglipAsTextEncoder, out_dir: str, filename: str = "siglip_adapter.pt"):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "hidden_proj": adapter.hidden_proj.state_dict(),
        "pool_proj": adapter.pool_proj.state_dict(),
        "siglip_config": adapter.model.config.to_dict(),
        "projection_dim": adapter.config.projection_dim,
    }, os.path.join(out_dir, filename))


def maybe_init_b2(bucket_name: str, prefix: Optional[str] = None):
    key_id = os.environ.get("B2_KEY_ID") or os.environ.get("B2_APPLICATION_KEY_ID")
    app_key = os.environ.get("B2_APPLICATION_KEY")
    if not key_id or not app_key or B2Api is None or InMemoryAccountInfo is None:
        return None, None

    info = InMemoryAccountInfo()
    api = B2Api(info)
    try:
        api.authorize_account("production", key_id, app_key)
        bucket = api.get_bucket_by_name(bucket_name)
    except Exception as e:  # pragma: no cover
        print(f"Warning: B2 init failed ({e}); disabling uploads.")
        return None, None

    if prefix is None:
        run_tag = None
        if wandb.run is not None:
            run_tag = f"{wandb.run.project}-{wandb.run.id}"
        if run_tag is None:
            run_tag = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"checkpoints/{run_tag}"
    return bucket, prefix


def b2_upload(bucket, local_path: str, remote_key: str):
    try:
        bucket.upload_local_file(local_file=local_path, file_name=remote_key)
        return True
    except Exception as e:  # pragma: no cover
        print(f"Warning: B2 upload failed for {local_path} -> {remote_key}: {e}")
        return False


def train():
    parser = argparse.ArgumentParser(description="SDXL baseline: use SigLIP (+ linear proj) as the sole text encoder or as a drop-in for CLIP-2")
    parser.add_argument("--pretrained_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--siglip_model", type=str, default="google/siglip-so400m-patch14-384")
    parser.add_argument("--output_dir", type=str, default="outputs/siglip-baseline")
    parser.add_argument(
        "--train_urls",
        type=str,
        default="https://f001.backblazeb2.com/file/ImageTrainingData/laion-pop-fixed/shard-{000000..000046}.tar",
        help="WebDataset shard pattern (default: 47 shards from 000000..000046 on Backblaze B2)",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_key", type=str, default="jpg")
    parser.add_argument("--text_key", type=str, default="txt")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"]) 
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    # wandb options
    parser.add_argument("--wandb_project", type=str, default="sdxl-siglip")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "online"), choices=["online", "offline", "disabled"]) 
    # image logging options
    parser.add_argument("--log_images_every", type=int, default=200, help="Log sample images to W&B every N steps; 0 disables")
    parser.add_argument("--sample_from_data_n", type=int, default=4, help="Number of (image,prompt) samples to draw from training set for logging")
    parser.add_argument("--sample_from_data_seed", type=int, default=12345)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--sample_height", type=int, default=768)
    parser.add_argument("--sample_width", type=int, default=768)
    parser.add_argument("--sample_seed", type=int, default=12345)
    # checkpointing / backups
    parser.add_argument("--save_every", type=int, default=1000, help="Save adapter checkpoint every N steps; 0 to disable periodic saves")
    parser.add_argument("--b2_bucket", type=str, default=os.environ.get("B2_BUCKET", "sdxl-siglip"))
    parser.add_argument("--b2_prefix", type=str, default=os.environ.get("B2_PREFIX"))
    parser.add_argument("--no_siglip", action="store_true", help="Disable SigLIP swap; train baseline SDXL text encoders")
    parser.add_argument("--safe_start_steps", type=int, default=50, help="Run first N steps with autocast disabled for extra numerical safety")
    parser.add_argument("--debug_log", action="store_true", help="Print debug stats (embeddings/noise preds/grad norms) periodically")
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
    if not args.no_siglip:
        # Single-encoder mode: remove TE1/TE2 and feed UNet with SigLIP projected to cross_attention_dim,
        # with sequence resampled to 77 tokens.
        pipe = swap_text_encoder_2_for_siglip(pipe, args.siglip_model, freeze_siglip=True, single_encoder=True)

    # Hard-freeze: UNet and VAE params (saves memory/compute)
    for p in pipe.unet.parameters():
        p.requires_grad = False
    for p in pipe.vae.parameters():
        p.requires_grad = False
    # SigLIP base is already frozen inside swap; keep adapter trainable
    pipe.unet.eval()
    pipe.vae.eval()
    if hasattr(pipe.text_encoder_2, "model"):
        pipe.text_encoder_2.model.eval()

    # Train only the projection layers (SigLIP adapter) when present
    if hasattr(pipe.text_encoder_2, "hidden_proj") and hasattr(pipe.text_encoder_2, "pool_proj"):
        params = list(pipe.text_encoder_2.hidden_proj.parameters()) + list(pipe.text_encoder_2.pool_proj.parameters())
    else:
        # Fallback: no-op tiny parameter to keep optimizer valid (should not be used in SigLIP mode)
        params = list(pipe.unet.parameters())[:0]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    # Scheduler: cosine with warmup
    warmup = max(10, min(500, args.max_steps // 10))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=args.max_steps)

    # Use the pipeline's scheduler/vae/unet directly
    noise_scheduler = pipe.scheduler
    vae = pipe.vae
    unet = pipe.unet

    # Data
    assert args.train_urls, "--train_urls must point to WebDataset shards"
    loader = make_wds_loader(
        args.train_urls,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_key=args.image_key,
        text_key=args.text_key,
    )

    unet, vae, optimizer = accelerator.prepare(unet, vae, optimizer)

    step = 0
    is_main = accelerator.is_main_process
    if is_main and args.wandb_mode != "disabled":
        mode = "disabled" if args.wandb_mode == "disabled" else args.wandb_mode
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=vars(args), mode=mode)

    # Draw a small fixed set of (prompt, training image) pairs from the dataset for logging comparisons
    compare_pairs: List[Dict[str, object]] = []
    if is_main and args.wandb_mode != "disabled" and args.log_images_every > 0:
        try:
            compare_pairs = sample_training_examples(
                args.train_urls,
                n=args.sample_from_data_n,
                seed=args.sample_from_data_seed,
                image_key=args.image_key,
                text_key=args.text_key,
            )
            # Log the raw training examples at step 0
            tbl = wandb.Table(columns=["prompt", "training_image"])
            for ex in compare_pairs:
                tbl.add_data(ex["prompt"], wandb.Image(ex["train_img"]))
            wandb.log({"samples/training_examples": tbl, "global_step": 0}, step=0)
        except Exception as e:
            accelerator.print(f"Warning: failed to sample training examples for logging: {e}")
    pipe.text_encoder_2.hidden_proj.train()
    pipe.text_encoder_2.pool_proj.train()
    unet.train()

    # B2 setup (main process only)
    b2_bucket = None
    b2_prefix = None
    if is_main:
        b2_bucket, b2_prefix = maybe_init_b2(args.b2_bucket, args.b2_prefix)

    # ETA tracking
    start_time = time.time()
    step_time_ema = None  # exponential moving average of step time

    for batch in loader:
        if step >= args.max_steps:
            break

        images, texts = batch  # images: [B,3,1024,1024], texts: list[str]

        with accelerator.accumulate(unet):
            t0 = time.time()
            images = images.to(device)

            # Latents (no grad for VAE encode)
            with torch.no_grad():
                latents = to_latents(vae, images, dtype=unet.dtype)
            noisy_latents, noise, timesteps = add_noise(latents, noise_scheduler)

            # Text conditioning (uses our swapped SigLIP encoder for pooled + seq embeds)
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encode_text(
                pipe, texts, device=device, guidance_scale=args.guidance_scale
            )

            # Debug: print embedding stats
            if is_main and args.debug_log and (step < 5 or step % 50 == 0):
                def stats(x):
                    x32 = x.detach().float()
                    return {
                        "shape": tuple(x32.shape),
                        "min": float(x32.min().item()),
                        "max": float(x32.max().item()),
                        "mean": float(x32.mean().item()),
                        "std": float(x32.std(unbiased=False).item()),
                        "norm": float(x32.norm().item()),
                    }
                accelerator.print(f"prompt_embeds: {stats(prompt_embeds)}")
                accelerator.print(f"pooled_prompt: {stats(pooled_prompt_embeds)}")

            # Sanity: ensure encoder_hidden_states have shape [B, T, C]
            assert prompt_embeds.dim() == 3, f"prompt_embeds dim {prompt_embeds.shape}"
            if negative_prompt_embeds is not None:
                assert negative_prompt_embeds.dim() == 3, f"neg_prompt_embeds dim {negative_prompt_embeds.shape}"

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
            use_fp32 = step < args.safe_start_steps
            autocast_ctx = torch.amp.autocast("cuda", enabled=(accelerator.mixed_precision != "no" and not use_fp32))
            if args.guidance_scale > 1.0:
                with autocast_ctx:
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
                with autocast_ctx:
                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids},
                    ).sample

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            if not torch.isfinite(loss):
                if is_main:
                    accelerator.print(f"Non-finite loss at step {step}; skipping optimizer step. loss={loss}")
                optimizer.zero_grad(set_to_none=True)
                step += 1
                continue

            accelerator.backward(loss)
            if is_main and args.debug_log and (step < 5 or step % 50 == 0):
                total_norm = 0.0
                for p in params:
                    if p.grad is not None:
                        total_norm += float(p.grad.detach().data.float().norm().item())
                accelerator.print(f"grad_norm (sum of norms): {total_norm:.6f}")
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            # Update EMA of step time
            dt = time.time() - t0
            step_time_ema = dt if step_time_ema is None else 0.9 * step_time_ema + 0.1 * dt

            if is_main and args.wandb_mode != "disabled" and (step % 10 == 0):
                remaining = max(args.max_steps - (step + 1), 0)
                eta_sec = (step_time_ema or dt) * remaining
                elapsed = time.time() - start_time
                # human-readable ETA
                hrs = int(eta_sec // 3600)
                mins = int((eta_sec % 3600) // 60)
                secs = int(eta_sec % 60)
                eta_str = f"{hrs:d}:{mins:02d}:{secs:02d}"
                wandb.log({
                    "train/loss": float(loss.detach().cpu()),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/step_time_s": dt,
                    "train/step_time_ema_s": float(step_time_ema or dt),
                    "train/eta_seconds": float(eta_sec),
                    "train/eta": eta_str,
                    "train/elapsed_s": float(elapsed),
                    "train/steps_per_sec_ema": (0.0 if (step_time_ema or dt) == 0 else 1.0 / float(step_time_ema or dt)),
                    "global_step": step,
                }, step=step)

            # Periodic local save + optional B2 upload
            if is_main and args.save_every > 0 and step > 0 and step % args.save_every == 0:
                ckpt_dir = os.path.join(args.output_dir, "adapter")
                fname = f"siglip_adapter_step{step}.pt"
                save_adapter(pipe.text_encoder_2, ckpt_dir, filename=fname)
                if b2_bucket is not None:
                    rel_key = f"{b2_prefix}/{fname}"
                    b2_upload(b2_bucket, os.path.join(ckpt_dir, fname), rel_key)

            # Periodically log sample images
            if is_main and args.wandb_mode != "disabled" and args.log_images_every > 0 and step % args.log_images_every == 0 and compare_pairs:
                prompts = [str(ex["prompt"]) for ex in compare_pairs]
                with torch.no_grad():
                    g = torch.Generator(device=device).manual_seed(args.sample_seed)
                    gen_images = pipe(
                        prompt=prompts,
                        num_inference_steps=args.sample_steps,
                        guidance_scale=args.guidance_scale,
                        height=args.sample_height,
                        width=args.sample_width,
                        generator=g,
                    ).images
                tbl = wandb.Table(columns=["prompt", "training_image", "generated_image"])
                for ex, gi in zip(compare_pairs, gen_images):
                    tbl.add_data(ex["prompt"], wandb.Image(ex["train_img"]), wandb.Image(gi))
                wandb.log({"samples/compare": tbl, "global_step": step}, step=step)

        if accelerator.is_main_process and step % 50 == 0:
            accelerator.print(f"step {step}: loss={loss.item():.4f}")

        step += 1

    if accelerator.is_main_process:
        # Final save
        ckpt_dir = os.path.join(args.output_dir, "adapter")
        final_name = "siglip_adapter.pt"
        save_adapter(pipe.text_encoder_2, ckpt_dir, filename=final_name)
        # Upload final
        if b2_bucket is not None:
            b2_upload(b2_bucket, os.path.join(ckpt_dir, final_name), f"{b2_prefix}/{final_name}")
        if args.wandb_mode != "disabled":
            wandb.save(os.path.join(args.output_dir, "adapter", "siglip_adapter.pt"))
            wandb.finish()

    accelerator.print("Training complete.")


if __name__ == "__main__":
    train()
