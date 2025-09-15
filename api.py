import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, StableDiffusionControlNetPipeline, ControlNetModel
import os
from safetensors.torch import load_file
from utils import fix_lora_keys
import io
import secrets
from typing import Optional
from PIL import Image

from fastapi import Depends, FastAPI, HTTPException, status, Form, Response, UploadFile, File
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from loguru import logger
from starlette.responses import StreamingResponse

app = FastAPI()

LORA_PATH = os.path.join("models", "lora-tsuki-epoch-20", "lora_adapter.safetensors")
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_ID = "lllyasviel/sd-controlnet-scribble" 

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
logger.info(f"Using device: {device}")


try:
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=torch.float16)
    
    scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        scheduler=scheduler,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    logger.info("Stable Diffusion pipeline loaded successfully")

    pipe_controlnet = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_ID, 
        scheduler=scheduler,
        controlnet=controlnet, 
        torch_dtype=dtype,
        safety_checker=None, 
        requires_safety_checker=False
    ).to(device)
    logger.info("Stable Diffusion with controlnet pipeline loaded successfully")

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    pipe_controlnet.enable_xformers_memory_efficient_attention()
    pipe_controlnet.enable_model_cpu_offload()
    
except Exception as e:
    logger.error(f"Failed to load Stable Diffusion pipeline: {e}")
    exit()

fixed_lora_dict = fix_lora_keys(LORA_PATH)
pipe.load_lora_weights(fixed_lora_dict)
pipe_controlnet.load_lora_weights(fixed_lora_dict)
logger.info("Load lora weights successfully")


@app.post("/inference")
async def inference(
    prompt: str = Form(""), 
    negative_prompt: str = Form(""),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(7.5),
    seed: int = Form(50)
):
    generator = torch.Generator(device=device).manual_seed(seed)
    logger.info(f"Generating image with seed: {seed}")

    if not prompt:
        prompt = " "

    try:
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                cross_attention_kwargs={"scale": 1.0}
            ).images[0]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
    
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        logger.error(f"An error occurred during inference: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/inference-scrible")
async def inference_scrible(
    prompt: str = Form(""), 
    negative_prompt: str = Form(""),
    scrible_image: UploadFile = File(...),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(7.5),
    seed: int = Form(50)
):
    scrible_image = Image.open(io.BytesIO(await scrible_image.read())).convert("RGB")

    generator = torch.Generator(device=device).manual_seed(seed)
    logger.info(f"Generating image with seed: {seed}")

    if not prompt:
        prompt = " "

    try:
        with torch.inference_mode():
            image = pipe_controlnet(
                prompt=prompt, 
                negative_prompt=negative_prompt,
                image=scrible_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                cross_attention_kwargs={"scale": 1},
                # controlnet_conditioning_scale=1
            ).images[0]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
    
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        logger.error(f"An error occurred during inference with controlnet: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/health")
def health_check():
    return {"status": "ok"}
    