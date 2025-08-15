#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复版应用
"""

import os
import sys

# 设置环境变量
os.environ["version"] = "v2ProPlus"
os.environ["is_half"] = "True"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meditation_app_fixed import MeditationApp
from GPT_SoVITS.inference_webui import i18n

print("\n测试修复版应用的生成功能...")

app = MeditationApp()

# 直接调用生成函数
result = app.generate_audio_simple(
    text="现在，让我们开始冥想。",
    ref_audio="downloads/test.mp3",
    ref_text="这是一段测试音频",
    preset_name="平静舒缓",
    speed=0.95,
    top_k=15,
    top_p=0.7,
    temperature=0.7,
    pause_second=0.4,
    text_language=i18n("中文"),
    ref_language=i18n("中文"),
    how_to_cut=i18n("按中文句号。切"),
    sovits_model=app.current_sovits,
    gpt_model=app.current_gpt
)

if result:
    sr, audio = result
    print(f"\n✅ 测试成功! 采样率={sr}, 音频长度={len(audio)/sr:.2f}秒")
else:
    print("\n❌ 测试失败!")