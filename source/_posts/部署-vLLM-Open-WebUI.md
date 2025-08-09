---
title: 部署:vLLM + Open WebUI
date: 2025-07-03 21:40:58
tags: [vllm, open_webui]
cover_image: /images/vllm.webp
cover: /images/vllm.webp
categories: NOTE
---
**创建虚拟环境：**
`conda create -n open-webui python==3.11`
**安装所有依赖：**

**`conda activate open-webui
pip install -U open-webui vllm torch transformers`**

**运行 vllm(使用微调后的模型)：**

**`vllm serve /root/autodl-tmp/code/work_dirs/qwen1_5_1_8b_chat_qlora_alpaca_e3/merged`**

**运行open-webui**
```zsh
export HF_ENDPOINT=https://hf-mirror.com
export ENABLE_OLLAMA_API=False 
export OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1
export DEFAULT_MODELS=
"/root/autodl-tmp/code/work_dirs/qwen1_5_1_8b_chat_qlora_alpaca_e3/merged"
open-webui serve --host 0.0.0.0 --port 8081
```
> 若卡住 手动添加端口进行转发 默认端口8080 