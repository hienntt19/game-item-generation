# Inference worker

## Introduction
As a key component of the Item Generation System, this inference worker is designed to consume messages from a RabbitMQ queue. It generates images based on the provided prompts and parameters, leveraging a fine-tuned LoRA with Stable Diffusion 1.5 model. 

## Overall architecture
<p align="center">
  <img src="images/system_architecture.png" alt="Sample dataset">
</p>

**Main workflows:**
- Message Consumption: the inference worker establishes a persistent connection to RabbitMQ, listening for incoming generation jobs on a dedicated queue.
- Inference and Image Generation: upon receiving a message, the worker uses the information (prompt, negative prompt, guidance scale, etc.) to generate image with the fine-tuned LoRA model.
- Status Updates and Storage: throughout the process, the worker communicates with an API Gateway (deployed on GKE) to update the job status in a Cloud SQL database:
  + When a job begins, its status is set to "Processing".
  + Upon successful completion, the status is updated to "Completed". The final image is uploaded to Google Cloud Storage (GCS), and its public URL is save to the database.
  + If an error occurs, the status is marked as "failed".

## Project structure
```
.
├── data_preparation      - Scripts for data crawling, processing and captioning
├── images                - Sample images
├── models                - Folder to save the downloaded SD1.5 model and fine-tuned LoRA
├── Dockerfile            - Defines the Docker image for the inference worker image
├── download_models.py    - Script to download the base SD1.5 model and fine-tuned LoRA weights
├── export_env   .sh      - Script to export environment variables
├── gcs_key.json          - GCS service account key
├── inference.py          - The core application logic for the worker
├── lora-fine-tune.ipynb  - Jupyter notebook for model fine-tuning
├── requirements.txt      - Python dependencies
└── setup_vm.sh           - Script to setup VM
```

## Table of content
- [Introduction](#introduction)
- [Overall architecture](#overall-architecture)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
1. [Model training Details](#1-model-training-details)
   1. [Data preparation](#11-data-preparation)
   2. [LoRA Fine-tuning](#12-lora-fine--tuning)
2. [Model serving Details](#2-model-serving-details)
   1. [VM Setup](#21-vm-setup)
   2. [Running the worker](#22-running-the-worker)

## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.11
- Uv
- Docker and Docker compose
- NVIDIA GPT with CUDA Toolkit 11.8
- Access credentials for RabbitMQ, Google Cloud Storage, API Gateway

## Environment Variables
| Variable                  | Description                                                                 
| ------------------------- | --------------------------------------------------------------------------- 
| `RABBITMQ_HOST`           | The hostname or IP address of the RabbitMQ server.                          
| `RABBITMQ_DEFAULT_USER`   | The username for RabbitMQ authentication.                                   
| `RABBITMQ_DEFAULT_PASS`   | The password for RabbitMQ authentication.                                   
| `API_GATEWAY_URL`         | The base URL of the API Gateway for status updates.                         
| `GCS_BUCKET_NAME`         | The name of the Google Cloud Storage bucket to upload generated images to.  
| `GOOGLE_APPLICATION_CREDENTIALS` | The path *inside the container* to the GCS key. Should be `/app/gcs_key.json`.


## 1. Model training Details
The core objective of the model training phase was to fine-tune a Stable Diffusion 1.5 model using LoRA. The goal was to enable the model to generate items that emulate the distinct art style of the Tsuki Adventure game assets (cute, hand-drawn, pastel colors, thick outlines, etc.)

### 1.1 Data preparation: Image crawling, processing and captioning
There are 3 main steps in this phase:
1. Image crawling
   1028 uniqe game asset images were scraped from the official Tsuki Adventure Fandom Wiki (https://tsuki-adventure.fandom.com/wiki/Items). This process was automated using python scipts leveraging Beautiful Soup and Selenium.

2. Image processing
   Images are converted to white background in this process. Below are some examples from the processed dataset.
   <p align="center">
   <img src="images/sample_dataset.png" alt="Sample dataset">
   </p>

3. Image captioning: Each image was then captioned to provide textual descriptions for the model. A two-stage captioning process was employed:
   - Automated captioning: initial captions were generated using the Google Gemini API.
   - Manual refinement: These automated captions were then manually reviewed and refined to ensure accuracy, consistency and detail.

   All captions adhere to a predefined structure to optimize the LoRA training, including a trigger word "tsuki_advtr", a description of the object and keywords defining the style.

   <p align="center">
   <img src="images/data_captioning.png" alt="Sample caption">
   </p>

### 1.2 LoRA Fine-tuning
The fine-tuning process was conducted within a Kaggle Notebook environment, utilizing a free T4 GPU instance.
   - Experiment tracking: Wandb was integrated for experiment tracking, which logging hyperparameters, training metrics, and final model artifacts.
   - Configuration: all settings and implementation details are in lora-fine-tune.ipynb notebook.

Example generation result:
   - prompt: "tsuki_advtr, a samoyed dog smiling, white background, thick outlines, pastel color, cartoon style, hand-drawn, 2D icon, game item, 2D game - style, minimalist", num_inference_steps: 50, guidance_scale: 7.5, seed: 50

   <p align="center">
   <img src="images/generated_image.png" alt="Sample image">
   </p>


## 2. Model serving Details
To serve the fine-tuned model, the inference worker is deployed on a cloud VM with GPU acceleration.

### 2.1 VM Setup
- Provider: Vast.ai (for cost-effective GPU rent options).
- Some specifications:
   + GPU: NVIDIA RTX 3060
   + OS template: Ubuntu 22.04
   + CUDA version: >= 11.8

- Automated setup: to install required dependencies (CUDA Toolkit, Conda, etc.):

   ```
   bash setup_vm.sh
   ```

### 2.2 Running the Inference worker

1. Clone the repository:
   ```
   git clone https://github.com/hienntt19/game-item-generation.git

   cd game-item-generation
   ```

2. Setup the python environment:
   ```
   uv init

   uv venv 

   source .venv/bin/activate

   uv pip install -r requirements.txt
   ```

3. Configure environment variables
   Update export_env.sh with your own credentials then run:
   ```
   source export_env.sh
   ```
4. Add GCS Service account key:
   Place your GCS service account key JSON file in secrets/ with named gcs_key.json. 

5. Download models:
   Run the download script to download Stable Diffusion 1.5 and fine-tuned LoRA:
   ```
   python download_models.py
   ```
   The downloaded models are placed in models/ directory.


There are 2 ways to run the inference worker:
- Running locally
   Start the worker directly:
   ```
   python inference.py
   ```

- Using Docker
   + Build the docker image:
   ```
   docker build -t inference-worker:latest .
   ```

   + Run the docker container:
   ```
   docker run -d \
      --restart always \
      --gpus all \
      -v /home/user/game-item-generation/models:/app/models \
      -v /home/user/secrets/gcs_key.json:/app/gcs_key.json \
      -e RABBITMQ_HOST=$RABBITMQ_HOST \
      -e RABBITMQ_DEFAULT_USER=$RABBITMQ_DEFAULT_USER \
      -e RABBITMQ_DEFAULT_PASS=$RABBITMQ_DEFAULT_PASS \
      -e API_GATEWAY_URL=$API_GATEWAY_URL \
      -e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
      --name my-inference-worker \
      inference-worker:latest
   ```



