import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import os
from safetensors.torch import load_file
from utils import fix_lora_keys


LORA_PATH = os.path.join("models", "lora-tsuki-epoch-20", "lora_adapter.safetensors")
IMAGES_PATH = "images"
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

os.makedirs(IMAGES_PATH, exist_ok=True)


def main():
    if not os.path.exists(LORA_PATH):
        print(f"Lora file not found: {LORA_PATH}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading pipeline on {device}...")
    scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        scheduler=scheduler,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(50)
    
    print("Loading and fixing lora...")
    fixed_lora_dict = fix_lora_keys(LORA_PATH)
    pipe.load_lora_weights(fixed_lora_dict)
    print("Lora loaded with fixed keys")
    

    prompt = "tsuki_advtr, a brown long bread, white background, thick outlines, pastel color, cartoon style, hand-drawn, 2D icon, game item, 2D game style, minimalist"
    negative_prompt = "pattern, multiple, many, repeating, tiling, background objects, floating objects, confetti, debris, clones, crowd"
    
    print(f"Generating image...")
    print(f"Prompt: {prompt}")

    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator,
        cross_attention_kwargs={"scale": 1.0}
    ).images[0]
    
    output_path = os.path.join(IMAGES_PATH, "test_bagel.png")
    image.save(output_path)
    
    print(f"Image saved to: {output_path}")

if __name__ == "__main__":
    main()