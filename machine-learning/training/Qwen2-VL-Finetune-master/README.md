# Fine-tuning Qwen2-VL Series

This repository contains a script for training [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) and [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) with only using HuggingFace and [Liger-Kernel](https://github.com/linkedin/Liger-Kernel).

## Other projects

**[[Gemma3 Finetune]](https://github.com/2U1/Gemma3-Finetune)**

## Update

- [2025/03/04] Add Option for using liger kernel.
- [2025/02/18] 🔥Support mixed-modality dataset with zero3.
- [2025/02/05] Fixed code for properly use image.
- [2025/02/03] Support Liger-kernel for Qwen2.5-VL.
- [2025/02/03] 🔥Supports Qwen2.5-VL.
- [2025/01/24] Add option for using DoRA.
- [2025/01/24] Fix error in LoRA training.
- [2025/01/18] 🔥Supports mixed-modality data.
- [2025/01/11] Updated 8-bit training with ms_amp fp8 with opt_level O3.
- [2024/11/05] Add memory efficient 8-bit training.
- [2024/09/12] 🔥Now the model is trained using [Liger-Kernel](https://github.com/linkedin/Liger-Kernel).
- [2024/09/11] Supports setting different learning rates to projector and vision model.
- [2024/09/11] 🔥Supports multi-image and video training.

## Table of Contents

- [Fine-tuning Qwen2-VL Series](#fine-tuning-qwen2-vl-series)
  - [Other projects](#other-projects)
  - [Update](#update)
  - [Table of Contents](#table-of-contents)
  - [Supported Features](#supported-features)
  - [Docker](#docker)
  - [Installation](#installation)
    - [Environments](#environments)
    - [Using `environment.yaml`](#using-environmentyaml)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
    - [Full Finetuning](#full-finetuning)
    - [Full Finetuning with 8-bit](#full-finetuning-with-8-bit)
    - [Finetune with LoRA](#finetune-with-lora)
    - [Train with video dataset](#train-with-video-dataset)
      - [Merge LoRA Weights](#merge-lora-weights)
      - [Image Resolution for performance boost](#image-resolution-for-performance-boost)
      - [Issue for libcudnn error](#issue-for-libcudnn-error)
  - [Inference](#inference)
    - [Gradio Infernce (WebUI)](#gradio-infernce-webui)
  - [TODO](#todo)
  - [Known Issues](#known-issues)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)
  - [VLLM Deployment Examples](#vllm-deployment-examples)
    - [1. Load Merged Model from Local Directory](#1-load-merged-model-from-local-directory)
    - [2. Load Merged Model from Hugging Face Hub](#2-load-merged-model-from-hugging-face-hub)
    - [3. Load Base Model and Apply LoRA Adapters Dynamically](#3-load-base-model-and-apply-lora-adapters-dynamically)

## Supported Features

- Deepspeed
- LoRA/QLoRA
- Full-finetuning
- Enable finetuning `vision_model` while using LoRA.
- Disable/enable Flash Attention 2
- Multi-image and video training
- Training optimized with liger kernel

## Docker

To simplify the setting process for training, you could use the provided pre-build environments.<br>
The settings are done in the conda env named `train`.<br><br>
You could find more information about the image [here](https://hub.docker.com/repository/docker/john119/vlm/general).

```
docker pull john119/vlm:v1
docker run --gpus all -it -v /host/path:/docker/path --name vlm --ipc=host john119/vlm:v1 /bin/bash
```

## Installation

### Environments

- Ubuntu 22.04
- Nvidia-Driver 550.120
- Cuda version 12.4

Install the required packages using `environment.yaml`.

### Using `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate qwen2
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

**Note:** You should install flash-attn after installing the other packages.

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

**When using a multi-image dataset, the image tokens should all be `<image>`, and the image file names should have been in a list.**<br><br>
**Please see the example below and follow format your data.**

<details>
<summary>Example for single image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for multi image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": ["000000033471.jpg", "000000033472.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n<image>\nIs the perspective of the camera different?"
      },
      {
        "from": "gpt",
        "value": "Yes, It the perspective of the camera is different."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for video dataset</summary>

```json
[
  {
    "id": "sample1",
    "video": "sample1.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nWhat is going on in this video?"
      },
      {
        "from": "gpt",
        "value": "A man is walking down the road."
      }
    ]
  }
  ...
]
```

</details>
<br><br>

Adding the new domain-specific data on top of the general data from open-source data will enhance downstream capabilities while retaining the foundational skills. Of course, you can also choose to fine-tune solely on the new data based on your requirements.

## Training

**Note:** Deepspeed zero2 is faster than zero3, however it consumes more memory. Also, most of the time zero2 is more stable than zero3.<br><br>
**Tip:** You could use `adamw_bnb_8bit` for optimizer to save memory.

To run the training script, use the following command:

### Full Finetuning

```bash
bash scripts/finetune.sh
```

### Full Finetuning with 8-bit

```bash
bash scripts/finetune_8bit.sh
```

**You need to install [ms-amp](https://github.com/Azure/MS-AMP) to use this script.**<br>
This script will finetune the model with fp8 model dtype. If you run out of vram, you could use this.<br>
You can even use offloading with fp8 training. For detailed config, you could change the deepspeed config files.

### Finetune with LoRA

**Note:** Liger-kernel won't work with QLoRA. You need to disable to use QLoRA.<br>
If you want to train only the language model with LoRA and perform full training for the vision model:

```bash
bash scripts/finetune_lora.sh
```

If you want to train both the language model and the vision model with LoRA:

```bash
bash scripts/finetune_lora_vision.sh
```

**IMPORTANT:** If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together.

<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Qwen2-VL model. **(Required)**
- `--use_liger` (bool): Option for using liger kernel to save memory.
- `--output_dir` (str): Output directory for model checkpoints
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--freeze_llm` (bool): Option to freeze LLM (default: False).
- `--tune_merger` (bool): Option to tune projector (default: True).
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--vision_lr` (float): Learning rate for vision_model.
- `--merger_lr` (float): Learning rate for merger(projector).
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--fp16` (bool): Option for using fp16.
- `--image_min_pixels` (int): Option for minimum input tokens for image.
- `--image_max_pixles` (int): Option for maximum maximum tokens for image.
- `--video_min_pixels` (int): Option for minimum input tokens for video.
- `--video_max_pixles` (int): Option for maximum maximum tokens for video.
- `--image_resized_width` (int): Option for setting the width of the input image.
- `--image_resized_height` (int): Option for setting the height of the input image.
- `--video_resized_width` (int): Option for setting the width of the input video.
- `--video_resized_height` (int): Option for setting the height of the input video.
- `--lora_enable` (bool): Option for using LoRA.
- `--vision_lora` (bool): Option for including `vision_tower` in LoRA module. `lora_enable` should be `True` to use this option.
- `--use_dora` (bool): Option for using DoRA instead of LoRA. `lora_enable` should be `True` to use this option.
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 32K).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 128).
- `--lora_alpha` (int): LoRA alpha (default: 256).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`.

</details>

### Train with video dataset

You can train the model using a video dataset. You can set LoRA configs and use for LoRA too.<br>

```bash
bash scripts/finetune_video.sh
```

**Note:** When training with video, it just as multi-image so you should adjust the `max_pixels` for maximum resolution and `fps` based on the available VRAM.

If you run out of vram, you can use [zero3_offload](./scripts/zero3_offload.json) instead of [zero3](./scripts/zero3_offload.json).<br>
You could use [zero2_offload](./scripts/zero2_offload.json) for a bit faster training.

#### Merge LoRA Weights

```
bash scripts/merge_lora.sh
```

**Note:** Remember to replace the paths in `finetune.sh` or `finetune_lora.sh` with your specific paths. (Also in `merge_lora.sh` when using LoRA.)

#### Image Resolution for performance boost

The model supports a wide range of resolution inputs. By default, it uses the native resolution for input.
For better performance using native or higher pixel numbers are recommended, however it takes too much memory and computation time for large images. So you could adjust the pixel numbers for it.
The model splits the image into `token * 28 * 28` so you could just change the the token_num part in the script. <br>
For example:

```
--image_min_pixels $((256 * 28 * 28))
--image_max_pixels $((1280 * 28 * 28))
--video_min_pixels $((128 * 28 * 28))
--video_max_pixels $((768 * 28 * 28))
```

Besides you could directly set the image/video height and width to control over the memory.

```
--resized_height 448
--resized_width 448
```

These values will be rounded to the nearest multiple of 28.

#### Issue for libcudnn error

```
Could not load library libcudnn_cnn_train.so.8. Error: /usr/local/cuda-12.1/lib/libcudnn_cnn_train.so.8: undefined symbol: _ZN5cudnn3cnn34layerNormFwd_execute_internal_implERKNS_7backend11VariantPackEP11CUstream_stRNS0_18LayerNormFwdParamsERKNS1_20NormForwardOperationEmb, version libcudnn_cnn_infer.so.8
```

You could run `unset LD_LIBRARY_PATH` for this error.
You could see this [issue](https://github.com/andimarafioti/florence2-finetuning/issues/2)

## Inference

**Note:** You should use the merged weight when trained with LoRA.

### Gradio Infernce (WebUI)

1. Install gradio

```
pip install gradio
```

2. Launch app

```
python -m src.serve.app \
    --model-path /path/to/merged/weight
```

You can launch gradio based demo with this command. This can also set some other generation configs like `repetition_penalty`, `temperature` etc.

## TODO

- [x] Support for video data
- [x] Add demo for multi-image and video
- [x] Handle mixed-modality data in dataset and collator
- [x] Support Qwen2.5-VL
- [x] Monkey-patch liger-kernel for Qwen2.5-VL
- [x] Update the code base to the latest transformers.
- [ ] Add DPO
- [ ] Add GRPO

## Known Issues

- [libcudnn issue](#issue-for-libcudnn-error)

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star: and citing:

```bibtex
@misc{Qwen2-VL-Finetuning,
  author = {Yuwon Lee},
  title = {Qwen2-VL-Finetune},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/2U1/Qwen2-VL-Finetune}
}
```

## Acknowledgement

This project is based on

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): An amazing open-source project of LMM.
- [Mipha](https://github.com/zhuyiche/llava-phi): Open-source project of SMM with amazing capabilities.
- [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct): Awesome pretrained MLLM based on Qwen2.
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel): Collection of Tirton kernels designed specifically for LLM training.

## VLLM Deployment Examples

Here are examples of how to deploy the model using the `vllm/vllm-openai` Docker image.

### 1. Load Merged Model from Local Directory

This method uses a model that has already been merged (e.g., using `finalize.py`) and saved locally.

```bash
sudo docker run --runtime=nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.huggingface:/root/.huggingface \
  -v /path/to/your/local/merged/model:/model \
  -p 8001:8000 \
  --ipc=host \
  -e HUGGING_FACE_HUB_TOKEN=<your_hf_token> \
  vllm/vllm-openai:v0.8.4 \
  --model /model \
  --trust-remote-code \
  --max-model-len 4096 \
  --max-num-seqs 256
```
*Replace `/path/to/your/local/merged/model` with the actual path to your merged model directory.*
*Replace `<your_hf_token>` with your Hugging Face Hub token if needed.*

### 2. Load Merged Model from Hugging Face Hub

This method loads a pre-merged model directly from a Hugging Face Hub repository.

```bash
sudo docker run --runtime=nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.huggingface:/root/.huggingface \
  -p 8001:8000 \
  --ipc=host \
  -e HUGGING_FACE_HUB_TOKEN=<your_hf_token> \
  vllm/vllm-openai:v0.8.4 \
  --model ChunkrAI/chunkr-table-v1-qwen2_5VL-3B \
  --trust-remote-code \
  --max-model-len 4096 \
  --max-num-seqs 256
```
*Replace `<your_hf_token>` with your Hugging Face Hub token, especially if the repository is private.*

### 3. Load Base Model and Apply LoRA Adapters Dynamically

This method loads the base model and applies LoRA adapters at runtime.

**Note:** This approach might not be ideal for Qwen-VL models as VLLM's dynamic LoRA might not correctly handle the vision components compared to a pre-merged model.

```bash
sudo docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.huggingface:/root/.huggingface \
  -p 8001:8000 \
  --ipc=host \
  -e HUGGING_FACE_HUB_TOKEN=<your_hf_token> \
  vllm/vllm-openai:v0.8.4 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --trust-remote-code \
  --enable-lora \
  --lora-modules <adapter_name>=<your_lora_repo_id> \
  --max-lora-rank 64 \
  --max-model-len 4096 \
  --max-num-seqs 256
```
*Replace `<your_hf_token>` with your Hugging Face Hub token.*
*Replace `<adapter_name>` with a name for your adapter (e.g., `finetuned_adapter`).*
*Replace `<your_lora_repo_id>` with the Hugging Face Hub repository ID containing *only* the LoRA adapter weights (not the merged model), e.g., `ChunkrAI/chunkr-table-v1-qwen2_5VL-3B-LORA`.*
