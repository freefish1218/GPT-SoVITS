#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试冥想音频生成功能
"""

import os
import sys
import torch

# 设置环境变量
os.environ["version"] = "v2ProPlus"
os.environ["is_half"] = "True"

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 测试导入
print("1. 测试导入模块...")
try:
    from config import get_weights_names, name2gpt_path, name2sovits_path
    SoVITS_names, GPT_names = get_weights_names()
    print(f"   ✓ 找到 SoVITS 模型: {len(SoVITS_names)}个")
    print(f"   ✓ 找到 GPT 模型: {len(GPT_names)}个")
    
    from GPT_SoVITS.inference_webui import (
        get_tts_wav,
        change_sovits_weights,
        change_gpt_weights,
        device,
        is_half,
        i18n,
        dict_language
    )
    print("   ✓ 成功导入推理函数")
except ImportError as e:
    print(f"   ✗ 导入错误: {e}")
    sys.exit(1)

# 测试模型加载
print("\n2. 测试模型加载...")
if SoVITS_names and GPT_names:
    sovits_model = SoVITS_names[0]
    gpt_model = GPT_names[-1]
    print(f"   使用 SoVITS 模型: {sovits_model}")
    print(f"   使用 GPT 模型: {gpt_model}")
    
    try:
        change_sovits_weights(sovits_model)
        print("   ✓ SoVITS 模型加载成功")
    except Exception as e:
        print(f"   ✗ SoVITS 模型加载失败: {e}")
    
    try:
        change_gpt_weights(gpt_model)
        print("   ✓ GPT 模型加载成功")
    except Exception as e:
        print(f"   ✗ GPT 模型加载失败: {e}")

# 测试音频生成
print("\n3. 测试音频生成...")
print("   测试参数:")
print("   - 参考音频: test_ref.wav (需要提供)")
print("   - 参考文本: 这是一段测试音频")
print("   - 目标文本: 现在让我们开始冥想")
print("   - 语言: 中文")

# 检查是否有测试音频
test_audio_path = "test_ref.wav"
if not os.path.exists(test_audio_path):
    print(f"   ⚠ 未找到测试音频文件 {test_audio_path}")
    print("   请提供一个 3-10 秒的参考音频文件")
    # 列出可用的音频文件
    import glob
    wav_files = glob.glob("*.wav")[:5]
    if wav_files:
        print(f"   找到的 wav 文件: {wav_files}")
        test_audio_path = wav_files[0]
        print(f"   使用: {test_audio_path}")

if os.path.exists(test_audio_path):
    try:
        print(f"\n   开始生成音频...")
        result = get_tts_wav(
            ref_wav_path=test_audio_path,
            prompt_text="这是一段测试音频",
            prompt_language=i18n("中文"),
            text="现在让我们开始冥想",
            text_language=i18n("中文"),
            how_to_cut=i18n("按中文句号。切"),
            top_k=15,
            top_p=0.7,
            temperature=0.7,
            ref_free=False,
            speed=0.95,
            if_freeze=False,
            inp_refs=None,
            sample_steps=8,
            if_sr=False,
            pause_second=0.4
        )
        
        print(f"   get_tts_wav 返回类型: {type(result)}")
        
        # 处理生成器
        if hasattr(result, '__iter__'):
            print("   处理生成器结果...")
            audio_count = 0
            for sr, audio in result:
                audio_count += 1
                print(f"   ✓ 生成音频 #{audio_count}: 采样率={sr}, 形状={audio.shape if hasattr(audio, 'shape') else 'N/A'}")
                # 保存第一个音频
                if audio_count == 1:
                    import numpy as np
                    import scipy.io.wavfile as wavfile
                    output_file = "test_output.wav"
                    wavfile.write(output_file, sr, audio)
                    print(f"   ✓ 音频已保存到: {output_file}")
        else:
            print(f"   ✗ 意外的返回类型: {result}")
            
    except Exception as e:
        print(f"   ✗ 生成失败: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n测试完成!")