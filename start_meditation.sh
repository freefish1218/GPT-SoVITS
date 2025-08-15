#!/bin/bash
# 冥想音频生成应用启动脚本

echo "🧘 正在启动冥想音频生成专业版..."
echo "================================================"

# 检查uv环境
if ! command -v uv &> /dev/null; then
    echo "❌ 错误：未找到uv，请先安装uv"
    echo "安装指令：curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export version="v2ProPlus"
export is_half="True"
export is_share="False"
export infer_ttswebui="9873"

# 禁用 Gradio 分析功能以避免网络连接问题
export GRADIO_ANALYTICS_ENABLED="False"
export GRADIO_TELEMETRY_ENABLED="False"

# 检查必要的依赖
echo "📦 检查依赖..."
uv run python -c "import gradio" 2>/dev/null || {
    echo "❌ 未找到gradio，正在安装..."
    uv add gradio
}

uv run python -c "import torch" 2>/dev/null || {
    echo "❌ 未找到torch，请先安装PyTorch"
    echo "安装指令：uv add torch torchvision torchaudio"
    exit 1
}

# 检查应用文件是否存在
if [ ! -f "meditation_app.py" ]; then
    echo "❌ 错误：未找到 meditation_app.py 文件"
    exit 1
fi

# 清理缓存
echo "🧹 清理缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 启动应用
echo "🚀 启动应用..."
echo "================================================"
echo "访问地址: http://localhost:9873"
echo "按 Ctrl+C 停止应用"
echo "================================================"

# 使用 timeout 命令避免长时间挂起，添加错误处理
timeout 300 uv run python meditation_app.py || {
    echo "❌ 应用启动失败或超时"
    echo "💡 提示：如果是网络问题，请检查网络连接"
    exit 1
}