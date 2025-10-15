import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import os
from safetensors.torch import load_file
import pika
import json 
import time
import requests
from google.cloud import storage
import uuid
from dotenv import load_dotenv

load_dotenv()

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
RABBITMQ_USER = os.getenv("RABBITMQ_DEFAULT_USER")
RABBITMQ_PASS = os.getenv("RABBITMQ_DEFAULT_PASS")
QUEUE_NAME = "image_generation_queue"

API_GATEWAY_URL = os.getenv("API_GATEWAY_URL")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcs_key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/gcs_key.json"

LORA_PATH = os.path.join("models", "lora-tsuki-epoch-20", "lora_adapter.safetensors")
IMAGES_PATH = "images"
# MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_ID = "models/stable-diffusion-v1-5"

os.makedirs(IMAGES_PATH, exist_ok=True)

def update_status(request_id, status, image_url=None):
    if not API_GATEWAY_URL:
        print("API_GATEWAY_URL is not set. Skipping status update.")
        return
    
    endpoint = f"{API_GATEWAY_URL}/update_db/{request_id}"
    
    payload = {
        "status": status
    }
    if image_url:
        payload["image_url"] = image_url
    
    try:
        response = requests.put(endpoint, json=payload)
        response.raise_for_status()
        print(f"[{request_id}] - Status updated to '{status}'")
    except requests.exceptions.RequestException as e:
        print(f"[{request_id}] - Failed to update status: {e}")


def upload_to_gcs(source_file_path, request_id):
    if not GCS_BUCKET_NAME:
        print("GCS_BUCKET_NAME is not set. Skipping upload to GCS.")
        return None
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        destination_blob_name = f"generated/{request_id}.png"
        blob = bucket.blob(destination_blob_name)
        
        blob.upload_from_filename(source_file_path)
        
        print(f"[{request_id}] - File uploaded to {destination_blob_name} in bucket {GCS_BUCKET_NAME}. Public URL: {blob.public_url}")
        return blob.public_url
    
    except Exception as e:
        print(f"[{request_id}] - Failed to upload file to GCS: {e}")
        return None


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading pipeline on {device}...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    print("Loading and setting DDIMScheduler...")
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler

    pipe.to(device)

    print(f"Loading lora weights from: {LORA_PATH}")
    if not os.path.exists(LORA_PATH):
        print(f"CRITICAL: Lora file not found at {LORA_PATH} inside the container!")
        return
        
    pipe.load_lora_weights(LORA_PATH)
    print("Lora loaded successfully.")
    
    return pipe, device


def generate_and_upload_image(pipe, device, request_id, params):
    prompt = params.get("prompt")
    negative_prompt = params.get("negative_prompt")
    num_inference_steps = int(params.get("num_inference_steps", 50))
    guidance_scale = float(params.get("guidance_scale", 7.5))
    seed = int(params.get("seed", 50))
    
    if not prompt:
        raise ValueError("Prompt is required for image generation.")
    
    print(f"[{request_id}] - Generating image with seed: {seed}")
    print(f"[{request_id}] - Prompt: {prompt}")

    generator = torch.Generator(device=device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        cross_attention_kwargs={"scale": 1.0}
    ).images[0]
    
    local_output_path = os.path.join(IMAGES_PATH, f"{request_id}.png")
    image.save(local_output_path)
    
    print(f"[{request_id}] - Image saved locally to: {local_output_path}")
    
    image_url = upload_to_gcs(local_output_path, request_id)
    
    try: os.remove(local_output_path)
    except OSError as e: print(f"[{request_id}] - Error removing temporary file: {e}")
    return image_url


def on_message_callback(ch, method, properties, body, pipe, device):
    request_id = None
    
    try:
        message = json.loads(body.decode('utf-8'))
        request_id = message.get("request_id")
        params = message.get("params")
        
        if not request_id or not params: raise ValueError("Invalid message format.")
        
        print(f"[{request_id}] - Received message: {message}")
        
        update_status(request_id, "processing")
        
        image_url = generate_and_upload_image(pipe, device, request_id, params)
        
        if image_url:
            update_status(request_id, "completed", image_url=image_url)
        else:
            raise RuntimeError("Failed to generate or upload image.")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"[{request_id}] - Task completed successfully.")   
        
    except Exception as e:
        print(f"[{request_id}] - Error processing message: {e}")
        if request_id: update_status(request_id, "failed")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"[{request_id or 'Unknown'}] - Task failed and acknowledged.")


def main():
    pipe, device = load_model()
    if pipe is None: return
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=credentials, heartbeat=600))
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            channel.basic_qos(prefetch_count=1)
            on_message_with_args = lambda ch, method, properties, body: on_message_callback(ch, method, properties, body, pipe, device)
            channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message_with_args, auto_ack=False)
            print('--> Connected to RabbitMQ. Waiting for messages...')
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Connection to RabbitMQ failed: {e}. Retrying in 10 seconds...")
            time.sleep(10)
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Restarting consumer...")
            time.sleep(10)

if __name__ == "__main__":
    main()