from llamafactory.train.tuner import run_exp

import argparse
import yaml
import os
from loguru import logger
from pathlib import Path

DEFAULT_CONFIG_DICT = {
    'use_unsloth_gc': True, 
    'enable_liger_kernel': True, 
    'stage': 'sft', 
    'do_train': True, 
    'finetuning_type': 'full', 
    'deepspeed': 'examples/deepspeed/ds_z3_offload_config.json', 
    'dataset': 'Sky-T1-HF', 
    'template': 'qwen25', 
    'cutoff_len': 16384, 
    'max_samples': 1000000, 
    'overwrite_cache': True, 
    'preprocessing_num_workers': 16, 
    'logging_steps': 1, 
    'save_steps': 600, 
    'plot_loss': True, 
    'per_device_train_batch_size': 1, 
    'gradient_accumulation_steps': 12, 
    'learning_rate': 1e-05, 
    'num_train_epochs': 3.0, 
    'lr_scheduler_type': 'cosine', 
    'warmup_ratio': 0.1, 
    'bf16': True, 
    'ddp_timeout': 180000000, 
    'report_to': 'wandb', 
    }

def launch(args_dict: dict):
    run_exp(args_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the config file, if is provided, all other arguments will be ignored")
    parser.add_argument("--model_name_or_path", type=str, help="Model name or path")
    parser.add_argument("--run_name", type=str, help="Run name")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    
    args, unknown_args = parser.parse_known_args()
    if args.config:
        if not os.path.exists(args.config) or not os.path.isfile(args.config) or not args.config.endswith('.yaml'):
            raise ValueError(f"Invalid config file: {args.config}")
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        args = argparse.Namespace(**config)  
    else:
        config = DEFAULT_CONFIG_DICT
        required_args = ['model_name_or_path', 'run_name', 'dataset']
        for arg in required_args:
            if not getattr(args, arg):
                raise ValueError(f"Argument {arg} is required")
        if not getattr(args, 'output_dir'):
            args.output_dir = f"./result/model/{args.run_name}"
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
        
        extra_config = {}
        for i in range(0, len(unknown_args), 2):
            key = unknown_args[i].lstrip("--")
            value = unknown_args[i + 1] if i + 1 < len(unknown_args) else True
            extra_config[key] = value
        config.update(extra_config)
        args = argparse.Namespace(**config)
    
    args.output_dir = Path(args.output_dir).resolve()
    args_dict = vars(args)
    logger.info(f"Arguments: {args_dict}")
    os.chdir("LLaMA-Factory")
    launch(args_dict)