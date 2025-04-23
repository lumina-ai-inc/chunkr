import copy
import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from PIL import Image
import re

from .params import DataArguments
from .constants import *

def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_image_info(image_path, min_pixel, max_pixel, width, height):
    # Using this because of process_vision_info function
    # Need to fix this in the future


    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]

def get_video_info(video_path, min_pixels, max_pixels, width, height, fps):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    content = {
        "type": "video", 
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "fps": fps
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    return video_input[0], video_kwargs

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(video_file, self.video_min_pixel, self.video_max_pixel, self.video_resized_w, self.video_resized_h, self.data_args.fps)
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
            
            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
            
            elif DEFAULT_VIDEO_TOKEN in user_input:
                if "Qwen2.5" in self.model_id:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt', **video_kwargs)
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                else:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict
    

def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)
