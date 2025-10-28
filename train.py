# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
import json
import sys
import os
import random
from functools import partial
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from accelerate.utils import DistributedType
from data_mix import Mix_dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model
from transformers import Trainer, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
from model.geopixel import GeoPixelForCausalLM

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='')
    # GeoPixelModel arguments
    vision_pretrained: Optional[str] = field(default='facebook/sam2-hiera-large')
    train_mask_decoder: bool = True
    out_dim : int = 256
    ce_loss_weight : float = 1.0
    dice_loss_weight : float = 0.5
    bce_loss_weight : float = 2.0
    is_pretrained: bool = False

@dataclass
class DataArguments:
    data_path: str = field(
        default='data.txt', metadata={'help': 'Path to the training data.'})
    given_num: bool = False
    batch_size: int = 4
    resolution: int = 560
    hd_num: int = 18


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    max_length: int = field(
        default=8192,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    fix_sampler: bool = True
    auto_resume: bool = False
    resume_dir: Optional[str] = field(default=None)
    start_epoch : int = field(default=0)
    label_names: List[str] = field(default_factory=lambda: ['samples'])

@dataclass
class LoraArguments:
    lora_r: int = 8 
    lora_alpha: int = 16 
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        'attention.wqkv',
        'attention.wo',
        'feed_forward.w1',
        'feed_forward.w2',
        'feed_forward.w3',
    ])
    lora_weight_path: str = ''
    lora_bias: str = 'none'

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = [instance['samples'] for instance in instances]
        text_input, data_type = tuple(
            [instance[key] for instance in instances]
            for key in ('text_input', 'data_type'))
        if 'image' not in instances[0]:
            text_input = [instance['text_input'][0] for instance in instances]
        batch = dict(
            text_input=text_input,
            data_type=data_type,
        )
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            batch['image'] = images
            if 'masks' in instances[0]:
                batch['image_g'] = [instance['image_g'] for instance in instances]
                batch['ori_hw'] = [instance['ori_hw'] for instance in instances]
                batch['masks'] = [instance['masks'] for instance in instances]  
        return dict(samples=batch)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    rank0_print('Loading data...')
    if data_args.data_path.endswith('json'):
        train_json = json.load(open(data_args.data_path))
    elif data_args.data_path.endswith('txt'):
        train_json = {}
        with open(data_args.data_path) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            line = line.split(' ')
            with open(line[0]) as f:
                temp = json.load(f)
            if data_args.given_num:
                assert len(line) == 2
                num = int(float(line[1]) * 1000)
                if len(temp) > num:
                    temp = random.sample(temp, num)
                else:
                    ex_temp = []
                    for i in range(num - len(temp)):
                        ex_temp.append(random.choice(temp))
                    temp.extend(ex_temp)
            else:
                if len(line) == 2:
                    ratio = float(line[1])
                    new_len = int(len(temp) * ratio)
                    if ratio < 1:
                        temp = random.sample(temp, new_len)
                    elif ratio > 1:
                        ex_temp = []
                        for i in range(new_len - len(temp)):
                            ex_temp.append(random.choice(temp))
                        temp.extend(ex_temp)
            rank0_print(f'Load {len(temp)} samples from {line}')
            train_json[line[0]] = temp

    train_dataset = Mix_dataset(
        train_json,
        data_args.batch_size,
        resolution=data_args.resolution,
        hd_num=data_args.hd_num,
        local_rank=local_rank)
    print(str(len(train_dataset)) + ' samples are loaded')
    eval_dataset = None

    data_collator = DataCollatorForSupervisedDataset()
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        logs = {}
        if hasattr(outputs, 'ce_loss'):
            logs['ce_loss'] = outputs.ce_loss.detach().cpu().item()
        if hasattr(outputs, 'mask_bce_loss'):
            logs['mask_bce_loss'] = outputs.mask_bce_loss.detach().cpu().item()
        if hasattr(outputs, 'mask_dice_loss'):
            logs['mask_dice_loss'] = outputs.mask_dice_loss.detach().cpu().item()
        if hasattr(outputs, 'mask_loss'):
            logs['mask_loss'] = outputs.mask_loss.detach().cpu().item()

        self.log(logs)

        return (loss, outputs) if return_outputs else loss
    
def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    
    if getattr(training_args, 'deepspeed', None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    config.max_length = training_args.max_length

    geo_model_args = {
        "vision_pretrained": model_args.vision_pretrained,
        "train_mask_decoder": model_args.train_mask_decoder,
        "out_dim": model_args.out_dim,
        "ce_loss_weight": model_args.ce_loss_weight,
        "dice_loss_weight": model_args.dice_loss_weight,
        "bce_loss_weight": model_args.bce_loss_weight,
    }

    # initializing tokenizer
    rank0_print(f'initialing tokenizer from: {model_args.model_name_or_path}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side='right',
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    special_tokens = ['[SEG]','<p>', '</p>']
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    seg_token_idx,bop_token_idx, eop_token_idx = [
        tokenizer(token, add_special_tokens=False).input_ids[0] for token in special_tokens
    ]

    geo_model_args.update({
        "seg_token_idx" : seg_token_idx, # segmentation token index
        "bop_token_idx" : bop_token_idx, # begining of phrase token index
        "eop_token_idx" : eop_token_idx  # end of phrase token index
    })

    torch_dtype = torch.float32
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.half

    # Load model and tokenizer
    rank0_print(f'Load model from: {model_args.model_name_or_path}')
    model = GeoPixelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        **geo_model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.tokenizer = tokenizer

    if not model_args.is_pretrained:
        rank0_print(f'Initializing Vision Modules: {model_args.vision_pretrained}')
        model.model.initialize_geopixel_modules(model.config)
    
    if training_args.use_lora:

        for name, param in model.model.named_parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type='CAUSAL_LM',
        )

        model = get_peft_model(model, lora_config)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    
    if training_args.fix_vit:
        model.vit.requires_grad_(False)
    else:
        model.vit.requires_grad_(True)
        model.vit.vision_tower.vision_model.post_layernorm = torch.nn.Identity()

    if training_args.fix_sampler:
        model.vision_proj.requires_grad_(False)
    else:
        model.vision_proj.requires_grad_(True)    
    
    # make some modules trainable
    trainable_modules = ["output", "tok_embeddings", "sam_mask_decoder", "text_hidden_fcs"]
    for name, param in model.named_parameters():
        if any([ module in name for module in trainable_modules]):
            param.requires_grad = True
    
    model.resize_token_embeddings(len(tokenizer))
    if training_args.use_lora:
        model.print_trainable_parameters()
    model.to(torch.bfloat16)
    
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Data type: {param.dtype} | Trainable: {param.requires_grad} | Size: {param.size()}")

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args)
    transformers.processing_utils.logging.enable_progress_bar()
    
    # Start trainer
    trainer = CustomTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module, 
    )
    trainer.train()
    trainer.save_state() 

    global_step = trainer.state.global_step
    last_checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-last")
    os.makedirs(last_checkpoint_dir, exist_ok=True)

    trainer.model_wrapped.save_checkpoint(last_checkpoint_dir)
    trainer.save_model(last_checkpoint_dir)
    
    rank0_print(f"Final checkpoint saved at step {global_step} in 'checkpoint-last/'")

if __name__ == '__main__':
    train()
