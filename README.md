## Prerequisite
```shell
cd LLaMA-Factory
pip install -e .
pip install nvitop # nvitop -m
cd ..
```
## Launch Training (New Version)
- dataset
  - Sky-T1-HF
  - VD-QWQ-Clean-8k
  - VD-QWQ-Clean-16k
  - VD-DS-Clean-8k
  - VD-QWQ-Noisy-Small-16k
  - VD-QWQ-Noisy-Small-8k
```shell
CUDA_VISIBLE_DEVICES=1,2 torchrun --nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_addr 127.0.0.1 \
--master_port 29501 \
train.py \
--model_name_or_path Qwen/Qwen2.5-7B-Instruct \
--run_name qwen2_7B_full_sft_1e-5 \
--dataset Sky-T1-HF \
--push_to_hub \
--push_to_hub_organization secmlr

CUDA_VISIBLE_DEVICES=3,4 torchrun --nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_addr 127.0.0.1 \
--master_port 29500 \
train.py \
--model_name_or_path Qwen/Qwen2.5-7B-Instruct \
--run_name qwen2_7B_full_sft_1e-5 \
--dataset VD-QWQ-Clean-8k \
--push_to_hub \
--push_to_hub_organization secmlr

# dataset: LLaMA-Factory/data/dataset_info.json, refer to VD-QWQ-Clean-8k
```
model_name_or_path, run_name and dataset are required. Besides, if output_dir is not specified, the training loges will be save at ./result/{args.dataset}/{args.run_name}.

A default set of training parameters, called DEFAULT_CONFIG_DICT, are provided. You can pass arguments in the DEFAULT_CONFIG_DICT to overwrite them. You can also pass arguments that are not in DEFAULT_CONFIG_DICT but supported by LLaMA-Factory.

```python
DEFAULT_CONFIG_DICT = {
    'use_unsloth_gc': True,
    'enable_liger_kernel': True,
    'stage': 'sft',
    'do_train': True,
    'finetuning_type': 'full',
    'deepspeed': 'examples/deepspeed/ds_z3_offload_config.json',
    'dataset': 'Sky-T1-HF',
    'template': 'qwen',
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
    'push_to_hub': False,
    'hub_strategy': 'all_checkpoints',
}
```
for example, if you want to sft 32B model with deepspeed zero3 offload, you can use the following command:

```shell
CUDA_VISIBLE_DEVICES=3,4 torchrun --nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_addr 127.0.0.1 \
--master_port 29500 \
train.py \
--model_name_or_path Qwen/Qwen2.5-32B-Instruct \
--run_name qwen2_7B_full_sft_1e-5 \
--dataset VD-QWQ-Clean-8k \
--deepspeed examples/deepspeed/ds_z3_offload_config.json
```

You can still use the yaml files if you want (if is provided, all other arguments will be ignored). Here is an example:
```shell
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.run --nnodes 1 --node_rank 0 --nproc_per_node 1 --master_addr 127.0.0.1 --master_port 29501 train.py --config configs/qwen2_3B_full_sft.yaml
```

## Launch Training (Old Version)
```shell
cd LLaMA-Factory
# 32B
CUDA_VISIBLE_DEVICES=0,2,3,4 MASTER_PORT=29501 \
llamafactory-cli train ../configs/qwen2_full_sft.yaml
# mix data
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29500 \
llamafactory-cli train ../configs/vd_qwq_sky_qwen2_full_sft.yaml

# 3b
CUDA_VISIBLE_DEVICES=4,5 MASTER_PORT=29502 \
llamafactory-cli train ../configs/qwen2_3B_full_sft.yaml
# 7b
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29501 \
llamafactory-cli train ../configs/qwen2_7B_full_sft.yaml
# 7b coder
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29500 \
llamafactory-cli train ../configs/qwen2_coder_7B_full_sft.yaml

# lora for 32B
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29501 \
llamafactory-cli train ../configs/qwen2_32B_lora_sft.yaml

# repro pipeline
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29501 \
llamafactory-cli train ../configs/qwen2_my_full_sft.yaml

# our dataset 32B
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29500 \
llamafactory-cli train ../configs/vd_ds_qwen2_full_sft.yaml
# qwq
CUDA_VISIBLE_DEVICES=6,7 MASTER_PORT=29500 \
llamafactory-cli train ../configs/vd_qwq_qwen2_full_sft.yaml

# our dataset 7B
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29501 \
llamafactory-cli train ../configs/vd_ds_qwen2_7b_full_sft.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29501 \
llamafactory-cli train ../configs/qwen2_full_simpo.yaml

python gpu_monitor.py --interval 30 "MASTER_PORT=29501 llamafactory-cli train ../configs/vd_ds_qwen2_full_sft.yaml"
```


| Metric       | Sky-T1-32B-Preview | my-32B-240 | 32B  | my-7B-720 | 7B    | 3B   | QwQ  | ds-r1 | o1-preview |
|--------------|--------------------|------------|------|-----------|-------|------|------|-------|------------|
| Math500      | 87.6               | 86.4       | 82.2 | 65.2      | 76.4  | 63.8 | 90.8 |       | 81.4       |
| AIME2024     | 46.67              | 33.33      | 23.3 | 13.33     | 10.0  | 6.67 | 40.0 |       | 40.0       |
| GPQA-Diamond | 51.01              | 50.0       | 44.4 | 27.27     | 34.85 | 29.8 | 54.0 |       | 75.2       |