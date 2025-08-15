#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试冥想应用UI - 监控按钮点击
"""

import os
import sys

# 设置环境变量
os.environ["version"] = "v2ProPlus"
os.environ["is_half"] = "True"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TELEMETRY_ENABLED"] = "False"

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*60)
print("启动冥想音频生成应用 - 调试模式")
print("="*60)
print("请在浏览器中打开应用并点击生成按钮")
print("控制台将显示详细的调试信息")
print("="*60 + "\n")

try:
    from meditation_app import MeditationApp
    
    # 创建应用
    app = MeditationApp()
    
    # 创建界面
    interface = app.create_interface()
    
    # 启动应用
    interface.launch(
        server_name="0.0.0.0",
        server_port=9873,
        inbrowser=False,  # 不自动打开浏览器
        share=False,
        show_error=True,
        quiet=False  # 显示所有日志
    )
    
except KeyboardInterrupt:
    print("\n应用已停止")
except Exception as e:
    print(f"\n启动失败: {e}")
    import traceback
    traceback.print_exc()