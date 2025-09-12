import argparse
import torch
from diffusers import DiffusionPipeline
from train_baseline import swap_text_encoder_2_for_siglip


def main():
    p = argparse.ArgumentParser(description="Generate a sanity-check image with SDXL+SigLIP adapter")
    p.add_argument("--pretrained_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--siglip_model", default="google/siglip-so400m-patch14-384")
    p.add_argument("--prompt", default="a photo of a corgi wearing sunglasses, studio lighting, high detail")
    p.add_argument("--out", default="outputs/siglip-baseline/sample.png")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)

    pipe = swap_text_encoder_2_for_siglip(pipe, args.siglip_model, freeze_siglip=True)

    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=1024,
        width=1024,
    ).images[0]

    out_path = args.out
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    image.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

