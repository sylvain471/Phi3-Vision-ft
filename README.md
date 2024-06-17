# Fine-tuning Phi3-Vision

This repository contains a script for training the [Phi3-Vision model](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct).

## Table of Contents

- [Fine-tuning Phi3-Vision](#fine-tuning-phi3-vision)
  - [Installation](#installation)
    - [Using `requirements.txt`](#using-requirementstxt)
    - [Using `environment.yml`](#using-environmentyml)
  - [Model Download](#model-download)
  - [Usage](#usage)
  - [Arguments](#arguments)
  - [Dataset Preparation](#dataset-preparation)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Supported Features

- Flexible selection of LoRA target modules
- Deepspeed Zero-2
- Deepspeed Zero-3
- Gradient checkpointing
- LoRA
- QLoRA
- Disable/enable Flash Attention 2
- Fine-tuning `img_projector` and `vision_tower` simultaneously
- Full-finetuning

## Installation

Install the required packages using either `requirements.txt` or `environment.yml`.

### Using `requirements.txt`

```bash
pip install -r requirements.txt
```

### Using `environment.yml`

```bash
conda env create -f environment.yml
conda activate phi3v
```

## Model Download

Before training, download the Phi3-V model from HuggingFace. It is recommended to use the `huggingface-cli` to do this.

1. Install the HuggingFace CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

2. Download the model:

```bash
huggingface-cli download microsoft/Phi-3-vision-128k-instruct --local-dir Phi-3-vision-128k-instruct --resume-download
```

## Usage

To run the training script, use the following command:

### Full Finetuning

```bash
bash scripts/finetune.sh
```

### Finetune with LoRA

If you want to train with LoRA:

```bash
bash scripts/finetune_lora.sh
```

#### Merge LoRA Weights

```
python merge_lora_weights.py \
    --model-path /Your/path/to/saved/weights \
    --model-base microsoft/Phi-3-vision-128k-instruct \
    --save-model-path /Your/path/to/save
```

**Note:** Remember to replace the paths in `finetune.sh` or `finetune_lora.sh` with your specific paths.

## Arguments

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Phi3-vision model. **(Required)**
- `--output_dir` (str): Output directory for model checkpoints (default: "output/test_train").
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False)
- `--tune_img_projector` (bool): Option to finetune img_projector (default: True)
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--non_lora_lr` (float): Learning rate for non lora modules.
- `--learning_rate` (float): Learning rate for lora_modules.
- `--bf16` (bool): Option for using bfloat16.
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 3072).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 128).
- `--lora_alpha` (int): LoRA alpha (default: 256).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.

<details>
<summary>Example Dataset</summary>

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

## Inference with CLI

```
python cli.py \
 --model-path /path/to/merged/weight \
 --image-file /Path/to/image/
```

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository userful in your project, please consider giving a star and citing:

```bibtex
@misc{phi3vfinetuning2023,
  author = {Gai Zhenbiao and Shao Zhenwei},
  title = {Phi3V-Finetuning},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/GaiZhenbiao/Phi3V-Finetuning},
  note = {GitHub repository},
}

@misc{phi3-vision-ft,
  author = {Yuwon Lee},
  title = {Phi-3-vision-ft},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/2U1/Phi3V-Finetuning},
  note = {GitHub repository, forked from \cite{phi3vfinetuning2023}},
}
```

## Acknowledgement

This project is based on

- [LLaVA](https://github.com/haotian-liu/LLaVA): An amazing open-source project of LMM.
- [Mipha](https://github.com/zhuyiche/llava-phi): Open-source projcet of SMM with amazing capabilites.
- [Microsoft Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct): Awesome pretrained SMM using phi3.
- [Phi3V-Finetuning](https://github.com/GaiZhenbiao/Phi3V-Finetuning): Open-source project for finetuning phi-3-vision.
