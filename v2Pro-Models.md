# GPT-SoVITS v2Pro 模型配置文档

## 概述
GPT-SoVITS v2Pro 版本是在保持 v2 硬件成本和推理速度优势的基础上，提供超越 v4 性能的语音合成模型。

## 性能特点
- **RTF推理速度**: 4060Ti (0.028) | 4090 (0.014) | M4 CPU (0.526)
- **显存占用**: 相比v2稍微增加，但保持高效率
- **音质**: 超过v4版本，特别适合平均音质较低的训练集

## 必需模型文件 (总计 ~4.4GB)

### v2Pro 核心模型 (615MB)
位置: `GPT_SoVITS/pretrained_models/v2Pro/`
```
s2Gv2Pro.pth        - SoVITS生成器 (162MB)
s2Dv2Pro.pth        - SoVITS判别器 (126MB)
s2Gv2ProPlus.pth    - 增强版生成器 (200MB)
s2Dv2ProPlus.pth    - 增强版判别器 (126MB)
```

### 共享基础模型 (3.2GB)
位置: `GPT_SoVITS/pretrained_models/`
```
s1v3.ckpt                                - GPT文本到语义模型
chinese-hubert-base/                     - 中文语音特征提取
chinese-roberta-wwm-ext-large/           - 中文文本编码
sv/pretrained_eres2netv2w24s4ep4.ckpt   - 说话人验证模型 (108MB)
```

### 语言支持模型 (622MB)
```
GPT_SoVITS/text/G2PWModel/              - 中文拼音转换 (561MB)
.venv/nltk_data/                        - 自然语言处理数据 (9MB)
.venv/lib/python3.12/site-packages/pyopenjtalk/open_jtalk_dic_utf_8-1.11/ - 日语字典 (22MB)
```

## 版本配置差异

### v2Pro 标准版
- 配置文件: `GPT_SoVITS/configs/s2v2Pro.json`
- 使用模型: `s2Gv2Pro.pth` + `s2Dv2Pro.pth`
- `upsample_initial_channel`: 512

### v2ProPlus 增强版  
- 配置文件: `GPT_SoVITS/configs/s2v2ProPlus.json`
- 使用模型: `s2Gv2ProPlus.pth` + `s2Dv2ProPlus.pth`
- `upsample_initial_channel`: 768
- 更高音质，需要更多显存

## 快速安装
```bash
# 创建虚拟环境
uv venv && source .venv/bin/activate

# 安装依赖
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install -r extra-req.txt --no-deps
uv pip install -r requirements.txt

# 使用MCP工具下载模型（推荐）
# 或者手动从 https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained 下载
```

## 音频参数
- 采样率: 32000 Hz
- 音频通道: 128  
- 段长度: 20480
- 支持语言: 中文、英语、日语、韩语、粤语

## 使用建议
- **v2Pro**: 适合一般用户，平衡性能与资源消耗
- **v2ProPlus**: 追求极致音质，显存充足的用户
- 特别适用于训练集音质一般但需要高质量输出的场景

## 文件验证
确保以下关键文件存在:
```bash
ls GPT_SoVITS/pretrained_models/v2Pro/s2*v2Pro*.pth
ls GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt
ls GPT_SoVITS/pretrained_models/s1v3.ckpt
ls GPT_SoVITS/text/G2PWModel/
```

---
*生成时间: 2025-08-15*
*模型来源: lj1995/GPT-SoVITS + XXXXRT/GPT-SoVITS-Pretrained*