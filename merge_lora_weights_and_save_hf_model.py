import sys
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from model.geopixel import GeoPixelForCausalLM


def parse_args(args):
    parser = argparse.ArgumentParser(description="merge lora weights and save model with hf format")
    parser.add_argument( "--version", default="MBZUAI/GeoPixel-7B")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--vision_pretrained", default='facebook/sam2-hiera-large', type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default=[
        'attention.wqkv',
        'attention.wo',
        'feed_forward.w1',
        'feed_forward.w2',
        'feed_forward.w3',
    ], type=list)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--weight", default="", type=str, required=True)
    parser.add_argument("--save_path", default="GeoPixel-7B", type=str)
    return parser.parse_args(args)

def main(args):
    args = parse_args(args)

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    tokenizer.pad_token = tokenizer.unk_token
    special_tokens = ['[SEG]','<p>', '</p>']
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    args.seg_token_idx,args.bop_token_idx, args.eop_token_idx = [
        tokenizer(token, add_special_tokens=False).input_ids[0] for token in special_tokens
    ]

    model_args = {
        "vision_pretrained": args.vision_pretrained,
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "seg_token_idx": args.seg_token_idx,
        "bop_token_idx" : args.bop_token_idx, 
        "eop_token_idx" : args.eop_token_idx  # end of phrase token index
    }

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model = GeoPixelForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.model.initialize_geopixel_modules(model.model.config)

    lora_r = args.lora_r
    if lora_r > 0:
        for _ , param in model.model.named_parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM',
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))
    model.to(torch_dtype)

    print(f"Loading model weights from: {args.weight} ...")
    state_dict = torch.load(args.weight, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    
    print("Merging adapter layers ...")
    model = model.merge_and_unload()

    print("Saving pretrained model and tokenizer ...")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    print("Model merged and saved successfully.")

if __name__ == "__main__":
    main(sys.argv[1:])
