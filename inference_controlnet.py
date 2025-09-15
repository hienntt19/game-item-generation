from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, DDIMScheduler
from controlnet_aux import HEDdetector
from utils import fix_lora_keys
import os
import torch
from safetensors.torch import load_file
from PIL import Image


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5" 
LORA_PATH = os.path.join("models", "lora-tsuki-epoch-20", "lora_adapter.safetensors")
CONTROLNET_ID = "lllyasviel/sd-controlnet-scribble" 
IMAGES_PATH = "images"

def main():
    scribble_image_path = os.path.join(IMAGES_PATH, "scribles", "bread.jpeg")
    control_image = Image.open(scribble_image_path).convert("RGB")
    
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=torch.float16)
    
    scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_ID, 
        scheduler=scheduler,
        controlnet=controlnet, 
        safety_checker=None, 
        torch_dtype=torch.float16
    ).to("cuda")
    
    pipe.requires_safety_checker = False
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    
    fixed_lora_dict = fix_lora_keys(LORA_PATH)
    pipe.load_lora_weights(fixed_lora_dict)
    
    generator = torch.Generator(device="cuda").manual_seed(50)
    
    
    prompt = "tsuki_advtr, a brown bread, white background, thick outlines, pastel color, cartoon style, hand-drawn, 2D icon, game item, 2D game style, minimalist"
    negative_prompt = "pattern, multiple, many, repeating, tiling, background objects, floating objects, confetti, debris, clones, crowd"
    
    image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        image=control_image,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator,
        cross_attention_kwargs={"scale": 1},
        # controlnet_conditioning_scale=1
    ).images[0]

    output_path = os.path.join(IMAGES_PATH, "test_scrible.png")
    image.save(output_path)
    
    print(f"Image saved to: {output_path}")
    
if __name__ == "__main__":
    main()