import copy
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
import ujson as json
from PIL import Image
from torch.utils.data import Dataset
# If you get rid of AutoProcessor, the code dosen't work.
from transformers import AutoProcessor

from .params import DataArguments

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100
LLaVA_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_TOKEN = "<|image_1|>"

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        padding=True,
    ):
        super(LazySupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        # rank0_print("Formatting inputs...Skip in lazy mode")
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        processor = self.processor
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder

            if not os.path.exists(image_file):
                image_file = os.path.join(image_folder, image_file)
            image = [Image.open(image_file).convert("RGB")]
        else:
            image = None
        sources = copy.deepcopy([e["conversations"] for e in sources])
        for i in range(len(sources)):
            sources[i] = llava_to_openai(sources[i])

        prompt = processor.tokenizer.apply_chat_template(sources[0], tokenize=False)

        prompt += processor.tokenizer.eos_token

        data_dict = processor(prompt, image, return_tensors="pt")

        if self.padding:
            # data_dict = processor.tokenizer.pad(
            #     data_dict,
            #     padding="max_length",
            #     max_length=training_length,
            #     return_tensors="pt",
            # )
            if 'pixel_values' not in data_dict:
                data_dict['pixel_values'] = torch.zeros([1, 17, 3, 336, 336], dtype=torch.bfloat16)
                data_dict['image_sizes'] = torch.zeros([1, 2], dtype=torch.int64)
            
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                attention_mask=data_dict["attention_mask"][0],
                pixel_values=data_dict["pixel_values"][0],
                image_sizes=data_dict["image_sizes"][0],
                labels=data_dict["labels"][0],
                # labels=processor.tokenizer.pad(
                #     {"input_ids": data_dict["labels"][0]},
                #     padding="max_length",
                #     max_length=training_length,
                #     return_tensors="pt",
                # ).input_ids,
            )
        else:
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                attention_mask=data_dict["attention_mask"][0],
                pixel_values=data_dict["pixel_values"][0],
                image_sizes=data_dict["image_sizes"][0],
                labels=data_dict["labels"][0],
            )
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        pixel_values = [instance["pixel_values"] for instance in instances]
        pixel_values = torch.stack(pixel_values, dim=0)
        image_sizes = [instance["image_sizes"] for instance in instances]
        image_sizes = torch.stack(image_sizes, dim=0)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        return batch
    

def llava_to_openai(data):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for entry in data:
        transformed_entry = {
            "role": role_mapping.get(entry["from"], entry["from"]),
            "content": entry["value"].replace(LLaVA_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN),
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def make_supervised_data_module(processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = LazySupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)