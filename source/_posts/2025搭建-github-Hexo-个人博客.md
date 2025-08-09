---
title: 2025搭建 github + Hexo 个人博客
date: 2025-07-02 22:35:08
tags: Hexo
categories: tutorial
cover: /images/image-20250702_cover.png
cover_image: /images/image-20250702_cover.png
cover_image_alt: “”
---

# 1.Hexo安装

#### 以macOS为例

- 安装Hexo：`npm install -g hexo-cli`

- 选好要安装的目录：`your/custom/path`,然后进入该目录`cd your/custom/path`

- 然后初始化Hexo：`hexo init blog`

- 进入blog文件夹: `cd blog`

- 安装依赖：`npm install`

- 启动服务：`hexo server`

- 访问：`localhost:4000`

  ![image-20250702213318274](/images/image-20250702213318274.png)

# 2.主题配置

> 主题：https://hexo.io/themes/

- 在`blog`目录下：

  - 首先：`git init`

  - 然后：`git submodule add https://github.com/Your/Hexo_Theme.git themes/Hexo_Theme_Name`

  - 修改`blog`目录下的`_config.yml`

    ``` yaml
    theme: Hexo_Theme_Name
    ```

# 3.部署到Github

- 修改`blog`目录下的`_config.yml`

```yaml
deploy:
  type: git
  # repo 建议使用ssh的方式 自行搜索或者AI 仓库提前创建，仓库名：your_github_username.github.io
  repo: git@github.com:your_github_username/your_github_username.github.io.git 
  branch: master
```

- 安装`deploy-git`: `npm install hexo-deployer-git --save`
- 然后：

```shell
hexo clean #清除之前生成的东西
hexo generate  #生成静态文章，缩写hexo g
hexo deploy  #部署文章，缩写hexo d
```

部署完成，访问`your_github_username.github.io`即可

- 使用自己的域名：`your_github_username.github.io -> Settings -> Pages -> Custom domain`

> 使用自己的域名首先解析一下域名

![image-20250702220239010](/images/image-20250702220239010.png)

使用`hexo d`部署完后域名会失效，需在`source`文件夹下新建一个`CNAME`文件，内容为你的域名。
