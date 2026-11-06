# Machine-learning · 机器学习课程设计

代码根目录：`D:\Machine-learning`。三个题目共用独立 Python 3.12 环境。
当前范围是配置环境、官方预训练权重和验收工具，尚未开展课程数据清洗、标注或训练。

## 本次验收结果（2026-10-26）

- PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128，RTX 4060 Laptop GPU 已实际完成 CUDA 运算及 GPU NMS。
- U-Net + ResNet34、YOLO11n-seg、YOLO11s-seg 的本地权重加载、128×128合成输入反向传播、512×512推理全部通过。
- `pip check` 通过；原始数据路径存在，未移动或改写原始数据。
- Excel内存读写、TF-IDF、jieba与Scikit-learn合成数据检查通过，见 `reports/common_dependencies_check.json`。
- 新环境禁用了用户级 site-packages，避免读入旧的 Python 3.13 用户包。
- PyCharm配置文件与SDK注册已检查；尚未通过PyCharm图形界面实际点击运行。

详细 GPU 验收记录见 `reports/environment_check.json`。这些是环境检查，不是课程模型精度或正式训练性能测试。

## 在 PyCharm 中打开

1. 选择 **File → Open**，打开 `D:\Machine-learning`，不要打开 `.venv` 子目录。
2. 项目解释器应显示 **Python 3.12 (Machine-learning)**。
3. 如当前 PyCharm 版本未自动识别，进入 **Settings → Project → Python Interpreter → Add Interpreter → Add Local Interpreter → Existing**，选择：
   `D:\Machine-learning\.venv\Scripts\python.exe`。
   无需新建第二个环境，也不要选 MSYS2、Anaconda base 或旧 Course_Design 环境。
4. 在运行配置选择 **Environment Check**，或直接运行 `scripts/verify_environment.py`。
5. Notebook 选择内核 **Python (Machine-learning)**。

PyCharm 2025.2 的 SDK 注册配置已在修改前备份。项目已排除环境、缓存和权重目录的索引。
配置文件检查不等于已在 PyCharm GUI 中实际点击验证；如界面未自动识别，按第3步指定解释器。

## 项目目录

| 目录 | 用途 |
|---|---|
| `task1_lane/` | 题目一：车道线失效预测代码 |
| `task2_bridge/` | 题目二：桥面裂缝分割代码 |
| `task3_intent/` | 题目三：中英文意图识别代码 |
| `src/coursework/` | 共享的环境设置、离线模型构建工具 |
| `configs/data_paths.toml` | 原始数据位置，只引用，不移动原文件 |
| `models/pretrained/` | 官方预训练权重，不纳入 Git |
| `models/manifest.json` | 权重来源、大小与 SHA256 |
| `data/processed/` | 后续处理数据存放位置 |
| `runs/` | 后续训练输出，当前不训练 |
| `reports/environment_check.json` | 最近一次环境验收结果 |
| `requirements.txt` | 直接依赖及版本约束 |
| `requirements-lock.txt` | 验收环境的完整版本记录 |

## 环境和模型

- Python 3.12.6，PyTorch 2.9.1 + CUDA 12.8，torchvision 0.24.1。
- SMP 0.5.0：U-Net + ResNet34，输出单通道裂缝 logits。ImageNet 编码器权重保存在本地。
- Ultralytics 8.3.161：YOLO11n-seg 和 YOLO11s-seg 的通用实例分割预训练权重。
- 表格任务：Pandas、NumPy、SciPy、Scikit-learn、imbalanced-learn、openpyxl。
- 文本任务：Scikit-learn 文本向量化、jieba、NLTK。NLTK 不附带额外下载语料；需要具体语料时再准备。
- 绘图与记录：Matplotlib、Seaborn、TensorBoard、JupyterLab、ipykernel。
- 图像处理：Pillow、OpenCV 和 torchvision.transforms.v2；暂不同时安装另一个 OpenCV headless 包。

这些权重不是已经训练好的桥梁裂缝模型。U-Net 解码器随机初始化；YOLO 通用类别需要在未来训练时适配。
YOLO 实例多边形与 U-Net 二值语义掩码不直接等价，正式训练前要确定标注转换规则。
原图先按物理桥段划分，避免相同位置重复照片跨集合，再切块和增强。

共享模型助手已作为本地可编辑包安装，PyCharm 也已将 `src` 设为 Sources Root。
在项目解释器或对应 Notebook 内核中可以直接使用：

```python
from coursework.models import build_unet, build_yolo

unet = build_unet()          # 本地编码器权重，不联网下载
yolo = build_yolo("n")      # 本地 YOLO11n-seg
```

## PowerShell 使用

不必激活环境，明确使用项目解释器即可：

```powershell
& 'D:\Machine-learning\.venv\Scripts\python.exe' 'D:\Machine-learning\scripts\verify_environment.py'
& 'D:\Machine-learning\.venv\Scripts\python.exe' -m pip check
```

如需重建环境，先用标准 Windows Python 3.12 创建 `.venv`，然后运行 `scripts/setup_environment.ps1`。
该脚本安装依赖、准备官方权重并运行合成输入验收，不启动课程训练。网络下载可能较大。
使用预编译 CUDA 包通常无须另外安装 CUDA Toolkit/cuDNN。不要直接使用系统默认 `pip`。
`requirements-lock.txt` 记录本次 Windows 环境；未来迁移 Linux 云服务器时应基于 `requirements.txt`
重新解析平台依赖，不能直接照搬 Windows 的 pywinpty 等包。

网络较慢时，`scripts/download_torch_wheel.py` 可分块续传当前 Windows CUDA 安装包，并验证官方 SHA256。
本次 PyTorch 大文件使用上海交大镜像传输，通用依赖使用阿里云 PyPI 镜像；模型权重来自 PyTorch 官方及 Ultralytics 官方 GitHub 发布。
下载工具和安装缓存位于 `.cache`，不会提交到 Git。

## 验收范围

`verify_environment.py` 检查依赖导入、CUDA矩阵运算、torchvision GPU NMS、三套模型的权重加载、
128×128随机输入反向传播、512×512随机输入推理。没有优化器更新，不读取课程图像，不保存训练权重。
合成损失仅验证梯度链路，并不验证正式分割损失、数据加载或精度。记录的显存只对应这些检查，不能用来推算正式训练批量。
正式训练前仍需完成标注规范、空间划分和数据流水线验证。
