#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试音频生成功能
"""

import os
import sys
import time

# 设置环境变量
os.environ["version"] = "v2ProPlus"
os.environ["is_half"] = "True"

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*60)
print("快速音频生成测试")
print("="*60)

# 导入应用
from meditation_app import MeditationApp
from GPT_SoVITS.inference_webui import i18n

# 创建应用实例
app = MeditationApp()

# 测试参数
test_audio = "downloads/test.mp3"
ref_text = "这是一段测试音频"
meditation_text = "现在，让我们开始今天的冥想练习。请找一个舒适的姿势坐下。"

print(f"\n测试参数:")
print(f"  参考音频: {test_audio}")
print(f"  参考文本: {ref_text}")
print(f"  目标文本: {meditation_text}")
print(f"  预设: 平静舒缓")

# 直接调用生成函数
print("\n开始生成音频...\n")

start_time = time.time()

try:
    result = app.generate_audio(
        text=meditation_text,
        ref_audio=test_audio,
        ref_text=ref_text,
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
        gpt_model=app.current_gpt,
        progress=None  # 不使用进度条
    )
    
    elapsed = time.time() - start_time
    
    if result:
        sr, audio = result
        print(f"\n✅ 音频生成成功!")
        print(f"  采样率: {sr} Hz")
        print(f"  音频长度: {len(audio)/sr:.2f} 秒")
        print(f"  生成耗时: {elapsed:.2f} 秒")
        
        # 保存音频
        import numpy as np
        import scipy.io.wavfile as wavfile
        output_file = "meditation_test_output.wav"
        wavfile.write(output_file, sr, audio)
        print(f"  已保存到: {output_file}")
        
    else:
        print(f"\n❌ 音频生成失败 (耗时 {elapsed:.2f} 秒)")
        
except Exception as e:
    print(f"\n❌ 发生错误: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("测试完成")
print("="*60)