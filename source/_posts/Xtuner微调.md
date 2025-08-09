---
title: Xtuner微调
date: 2025-07-03 14:57:15
tags: Xtuner
categories: NOTE
cover_image: /images/Blog-Banner.png
cover: /images/Blog-Banner.png
---
```jsx
github: https://github.com/InternLM/xtuner
```

# 安装

```jsx
conda create --name xtuner-env python=3.10 -y
conda activate xtuner-env

git clone https://github.com/InternLM/xtuner.git
cd xtuner
pip install -e '.[all]'
```

# 配置文件

目录：xtuner/xuner/configs/[对应的模型]/[对应的模型配置].py

配置内容（QLora为例）：

下载对应的模型：

```jsx
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen1.5-1.8B-Chat', cache_dir='autodl-tmp/models')
```

将模型配置文件复制到根目录（代码目录）：

修改参数：

```jsx
# Copyright (c) OpenMMLab. All rights reserved.
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (
    CheckpointHook,
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
from xtuner.engine.hooks import (
    DatasetInfoHook,
    EvaluateChatHook,
    VarlenAttnArgsToMessageHubHook,
)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.parallel.sequence import SequenceParallelSampler
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
**# Model 修改模型路径**
**pretrained_model_name_or_path = "/root/autodl-tmp/models/Qwen/Qwen1___5-1___8B-Chat"**
use_varlen_attn = False

# Data
**alpaca_en_path = "/root/autodl-tmp/code/data/train_data.json" # 自定的json**
prompt_template = PROMPT_TEMPLATE.qwen_chat
**max_length = 512**
pack_to_max_length = True

# parallel
sequence_parallel_size = 1

# Scheduler & Optimizer
**batch_size = 6**  # per_device
accumulative_counts = 16
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 0
**max_epochs = 10**
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
evaluation_freq = 500
**SYSTEM = '你由Veeblue打造的中文领域心理健康助手, 是一个研究过无数具有心理健康问题的病人与心理健康医生对话的心理专家, 在心理方面拥有广博的知识储备和丰富的研究咨询经验，你旨在通过专业心理咨询, 协助来访者完成心理诊断。请充分利用专业心理学知识与咨询技术, 一步步帮助来访者解决心理问题, 接下来你将只使用中文来回答和咨询问题。'
evaluation_inputs = [
'请介绍你自己', # self cognition
'你好',
'我今天心情不好，感觉不开心，很烦。',
'我最近总是感到很焦虑，尤其是在学业上。我有个特别崇拜的同学，他好像在各方面都比我优秀，我总觉得自己怎么努力也追不上他，这让我压力特别大。',
]**

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        **quantization_config=None,  # 设置None不启用QLora微调
        # 若启用QLora：
        # dict(
        #     type=BitsAndBytesConfig,
        #     load_in_4bit=True,
        #     load_in_8bit=False,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # ),**
    ),
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
alpaca_en = dict(
    type=process_hf_dataset,
    **dataset=dict(type=load_dataset, path='json', data_files=alpaca_en_path), #使用自定义的json**
    tokenizer=tokenizer,
    max_length=max_length,
    **dataset_map_fn=None,**
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn,
)

sampler = SequenceParallelSampler if sequence_parallel_size > 1 else DefaultSampler

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=alpaca_en,
    sampler=dict(type=sampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template,
    ),
]

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
visualizer = None

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)

```

其中**train_data.json：**

```jsx
[
    {
        "conversation": [
            {
                "input": "最近总是觉得自己很焦虑，感觉压力很大，怎么办呢？",
                "output": "我很明白你的感受，焦虑和压力都是很正常的情绪，但是我们可以一起想办法来缓解它们。"
            },
            {
                "input": "可是我总是感觉自己做的不够好，压力和焦虑情绪就会更加加重，怎么破？",
                "output": "别太苛责自己了，我们都是人，总有做得不够好的时候。也许你可以试着规划一下自己的计划，分解成小目标，逐步完成，让自己的进步更可见。"
            },
            {
                "input": "可是我总是会拖延，根本不能按照计划执行，怎么办呢？",
                "output": "是啊，拖延也是很常见的问题。你可以尝试用时间管理的方法，设定时间表和截止时间，提醒自己要按照计划去行动。"
            },
            ......
            
]
```

运行：

```jsx
xtuner train qwen1_5_1_8b_chat_qlora_alpaca_e3.py
```

> **报错：ModuleNotFoundError: No module named 'triton.ops’**
> 

**解决：**`pip install --upgrade bitsandbytes`

**后台训练**：使用`screen`命令开启，`ctl + a + d`会退出到原终端，并且显示detached，意味着这个会话只是离开并未退出。

**重进入会话：**`screen -ls`

![截屏2025-06-27 14.27.35.png](/images/%E6%88%AA%E5%B1%8F2025-06-27_14.27.35.png)

**恢复会话**：screen -r 【会话ID】

![截屏2025-06-27 14.29.09.png](/images/%E6%88%AA%E5%B1%8F2025-06-27_14.29.09.png)

**模型转换：**pth —> hf

**`xtuner convert pth_to_hf $CONFIG_NAME_OR_PATH $PTH $SAVE_PATH`**

`xtuner convert pth_to_hf ./qwen1_5_1_8b_chat_qlora_alpaca_e3.py ./iter_9560.pth ./hf`

**转换出错：**

```jsx
(xtuner-env) root@autodl-container-e1ee44a95d-c3292730:~/autodl-tmp/code/work_dirs/qwen1_5_1_8b_chat_qlora_alpaca_e3# xtuner convert pth_to_hf ./qwen1_5_1_8b_chat_qlora_alpaca_e3.py ./iter_9560.pth ./hf
[2025-06-28 13:28:45,006] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-06-28 13:28:49,879] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Traceback (most recent call last):
  File "/root/autodl-tmp/xtuner/xtuner/tools/model_converters/pth_to_hf.py", line 151, in <module>
    main()
  File "/root/autodl-tmp/xtuner/xtuner/tools/model_converters/pth_to_hf.py", line 123, in main
    state_dict = guess_load_checkpoint(args.pth_model)
  File "/root/autodl-tmp/xtuner/xtuner/model/utils.py", line 314, in guess_load_checkpoint
    state_dict = torch.load(pth_model, map_location="cpu")
  File "/root/miniconda3/envs/xtuner-env/lib/python3.10/site-packages/torch/serialization.py", line 1524, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
        (1) In PyTorch 2.6, we changed the default value of the weights_only argument in torch.load from False to True. Re-running torch.load with weights_only set to False will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        (2) Alternatively, to load with weights_only=True please check the recommended steps in the following error message.
        WeightsUnpickler error: Unsupported global: GLOBAL mmengine.logging.history_buffer.HistoryBuffer was not an allowed global by default. Please use torch.serialization.add_safe_globals([mmengine.logging.history_buffer.HistoryBuffer]) or the torch.serialization.safe_globals([mmengine.logging.history_buffer.HistoryBuffer]) context manager to allowlist this global if you trust this class/function.
Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
```

**原因：**

这个错误是由于 PyTorch 2.6 改变了 `torch.load` 的默认行为导致的。新版本默认启用了 `weights_only=True` 参数以提高安全性，但这导致某些包含非标准对象的检查点文件无法加载。

**解决：**

修改 xtuner 源码（推荐）找到错误提示中的文件 `/root/autodl-tmp/xtuner/xtuner/model/utils.py`，在第314行附近修改 `torch.load` 调用：

```jsx
# 原来的代码
state_dict = torch.load(pth_model, map_location="cpu")

# 修改为
state_dict = torch.load(pth_model, map_location="cpu", weights_only=False)
```

**再次执行：**

![截屏2025-06-28 13.43.38.png](images/%E6%88%AA%E5%B1%8F2025-06-28_13.43.38.png)

**模型合并：**

**`xtuner convert merge /root/autodl-tmp/models/Qwen/Qwen1___5-1___8B-Chat ./hf ./merged --max-shard-size 2GB`**

![截屏2025-06-28 13.47.24.png](/images/%E6%88%AA%E5%B1%8F2025-06-28_13.47.24.png)

**模型聊天：**

`xtuner chat ./merged --prompt-template qwen_chat`

> **`--prompt-template 参数选择:** choose from 'default', 'zephyr', 'internlm_chat', 'internlm2_chat', 'moss_sft', 'llama2_chat', 'code_llama_chat', 'chatglm2', 'chatglm3', 'qwen_chat', 'baichuan_chat', 'baichuan2_chat', 'wizardlm', 'wizardcoder', 'vicuna', 'deepseek_coder', 'deepseekcoder', 'deepseek_moe', 'deepseek_v2', 'mistral', 'mixtral', 'minicpm', 'minicpm3', 'gemma', 'cohere_chat', 'llama3_chat', 'phi3_chat’`