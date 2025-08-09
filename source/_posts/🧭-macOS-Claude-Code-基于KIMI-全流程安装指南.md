---
title: 🧭 macOS Claude Code(使用Kimi Api)全流程安装指南
date: 2025-07-19 10:48:23
tags: [claude, kimi, macOS]
categories: Tutorial
cover: /images/250719_cover.png
cover_image: /images/250719_cover.png
---

>以下操作默认你已具备🪄**科学上网**的条件以及已经安装了🍺`Homebrew`

### 🧩 安装Node.js

```shell
brew install node
```

### 🧠 安装Claude Code

```shell
npm install -g @anthropic-ai/claude-code
```

### 🔑 申请Kimi API Key

[🔗 点我直达申请](https://platform.moonshot.cn/console)

![iShot_2025-07-18_18.45.22](/images/iShot_2025-07-18_18.45.22.png)

![iShot_2025-07-18_18.45.58](/images/iShot_2025-07-18_18.45.58.png)

> 🔔保存好你的`Api Key`,页面只会显示一次！

### 💻 配置kimi api环境变量

- 打开终端输入：`vim ~/.zshrc`

- 把以下内容追加到`.zshrc`文件里：

```shell
export ANTHROPIC_BASE_URL="https://api.moonshot.cn/anthropic"
export ANTHROPIC_AUTH_TOKEN="你的Kimi API Key"
```

![iShot_2025-07-18_18.38.23](/images/iShot_2025-07-18_18.38.23.png)

### ⚙️ 配置Claude Code

- 打开终端，输入：`ll -a`

![iShot_2025-07-18_18.51.01](/images/iShot_2025-07-18_18.51.01.png)

- 输入命令：`cd .claude`进入`.claude`文件夹中

  > 也可以输入`open .claude`,窗口化打开`.claude`目录，然后可视化进行下续操作
  >
  > ![iShot_2025-07-18_19.14.48](/images/iShot_2025-07-18_19.14.48.png)

- 创建一个`settings.json`文件，输入命令：`vim settings.json`，输入`i`开启`-- INSERT --(插入模式)`，编辑好以下内容后粘贴进去，然后按键盘的`ESC`键，输入`:wq`保存退出.

```json
{
    "env":{
        "HTTP_PROXY": "http://127.0.0.1:7897",
        "HTTPS_PROXY": "http://127.0.0.1:7897"
    }
}
```

> 🌍 `7897`为端口号

- 端口号修改为你自己代理工具的端口号：

![iShot_2025-07-18_18.55.59](/images/iShot_2025-07-18_18.55.59.png)

- 最后把代理切换到**全局**：

![iShot_2025-07-18_18.59.21](/images/iShot_2025-07-18_18.59.21.png)

- 终端运行`claude`

![iShot_2025-07-18_19.03.05](/images/iShot_2025-07-18_19.03.05.png)

- 可以看到API已经是**月之暗面**(Kimi)的了
