# SuperPulsar User Manual

## 1. 项目背景

爱芯元智 AI 工具链使用手册公共维护项目

- 提供统一的 AI 工具链文档开发者社区展示地址
- 降低 AI 工具链 Developer 维护成本
- 降低 AI 工具链 User 学习成本

## 2. 本地运行指南

### 2.1 git clone & update submodule

注意: 开发机需要安装 Git LFS, 如果未安装可以通过以下命令安装

```bash
git lfs install
```

安装 lfs 服务后, 执行

```bash
git clone git@git-ext.axera-tech.com:npu/superpulsar-doc.git
# update ax-samples
git submodule update --init --recursive
```

目录树如下:

```bash
.
├── build
│   ├── doctrees
│   └── html
├── LICENSE
├── make.bat
├── Makefile
├── README.md
├── requirements.txt
├── source                     # 文档主体
│   ├── conf.py
│   ├── examples               # 以 .zip 格式保存了一些例子, 由于git pages的限制, 在线文档不支持点击下载操作
│   ├── faq
│   ├── index.rst
│   ├── media
│   ├── superpulsar
│   ├── test_configs
│   ├── user_guides_advanced
│   ├── user_guides_quick
│   └── user_guides_runtime
└── third_party
    ├── ax-samples             # 模型上板运行示例 Demo
    ├── README.md
    └── test-model-zoo         # 相关 onnx 以及对应的 joint 模型
```

### 2.2 编译

安装依赖

```bash
pip install -r requirements.txt
```

在项目根目录下执行以下命令

```bash
$ make clean
$ make html
```

### 2.3 预览

完成编译后，使用浏览器查看 `build/html/index.html` . 如果在服务器上开发, 可以通过 `ssh` 端口转发的方式访问编译后的文档, 方法如下:

首先可以利用 `python` 在编译后的 `build/html/` 文件夹下启动一个 `http` 服务,

```bash
$ cd build/html/
$ python -m SimpleHTTPServer 8005  # For python2, 端口可以自定义
# or
$ python3 -m http.server 8005      # For python3, 端口可以自定义
```

然后通过 `ssh` 链接服务器,

```bash
ssh -L 8005:localhost:8005 username@server
```

然后本地浏览器访问: `localhost:8005/index.html`

## 3. 参考

- 本项目基于 Sphinx 搭建，关于更多 Sphinx 的信息请见 https://www.sphinx-doc.org/en/master/

## 4. 发版


