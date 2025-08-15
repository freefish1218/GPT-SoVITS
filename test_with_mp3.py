#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用提供的MP3文件测试音频生成
"""

import os
import sys
import torch

# 设置环境变量
os.environ["version"] = "v2ProPlus"
os.environ["is_half"] = "True"

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("音频生成测试脚本")
print("=" * 60)

# 测试导入
print("\n1. 导入模块...")
try:
    from config import get_weights_names, name2gpt_path, name2sovits_path
    SoVITS_names, GPT_names = get_weights_names()
    print(f"   ✓ 找到 SoVITS 模型: {SoVITS_names[:2]}...")
    print(f"   ✓ 找到 GPT 模型: {GPT_names[:2]}...")
    
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

# 检查测试音频
test_audio_path = "downloads/test.mp3"
print(f"\n2. 检查测试音频: {test_audio_path}")
if os.path.exists(test_audio_path):
    print(f"   ✓ 文件存在，大小: {os.path.getsize(test_audio_path)} bytes")
else:
    print(f"   ✗ 文件不存在")
    sys.exit(1)

# 转换MP3为WAV
print("\n3. 转换MP3为WAV...")
try:
    import librosa
    import soundfile as sf
    import tempfile
    
    # 读取MP3
    print("   正在读取MP3...")
    audio_data, sr = librosa.load(test_audio_path, sr=None)
    print(f"   ✓ 音频采样率: {sr} Hz")
    print(f"   ✓ 音频长度: {len(audio_data)/sr:.2f} 秒")
    
    # 创建临时WAV文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        wav_path = tmp_wav.name
        sf.write(wav_path, audio_data, sr)
        print(f"   ✓ 已转换为WAV: {wav_path}")
except Exception as e:
    print(f"   ✗ 转换失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 加载模型
print("\n4. 加载模型...")
if SoVITS_names and GPT_names:
    sovits_model = SoVITS_names[0]
    gpt_model = GPT_names[-1]
    print(f"   使用 SoVITS: {sovits_model}")
    print(f"   使用 GPT: {gpt_model}")
    
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
print("\n5. 测试音频生成...")
print("   参数:")
print(f"   - 参考音频: {wav_path}")
print("   - 参考文本: 这是一段测试音频")
print("   - 目标文本: 现在，让我们开始今天的冥想练习。请找一个舒适的姿势坐下。")
print("   - 语言: 中文")

try:
    print("\n   开始生成...")
    result = get_tts_wav(
        ref_wav_path=wav_path,
        prompt_text="这是一段测试音频",
        prompt_language=i18n("中文"),
        text="现在，让我们开始今天的冥想练习。请找一个舒适的姿势坐下。",
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
    
    print(f"\n   返回类型: {type(result)}")
    
    # 处理生成器
    if hasattr(result, '__iter__'):
        print("   处理生成器结果...")
        audio_count = 0
        for sr, audio in result:
            audio_count += 1
            print(f"   ✓ 音频片段 #{audio_count}:")
            print(f"     - 采样率: {sr} Hz")
            print(f"     - 形状: {audio.shape if hasattr(audio, 'shape') else 'N/A'}")
            print(f"     - 类型: {type(audio)}")
            
            # 保存第一个音频
            if audio_count == 1:
                import numpy as np
                import scipy.io.wavfile as wavfile
                output_file = "test_meditation_output.wav"
                wavfile.write(output_file, sr, audio)
                print(f"   ✓ 音频已保存到: {output_file}")
                
                # 打印音频统计
                print(f"     - 最大值: {np.max(np.abs(audio))}")
                print(f"     - 时长: {len(audio)/sr:.2f} 秒")
    else:
        print(f"   ✗ 意外的返回类型: {result}")
        
except Exception as e:
    print(f"\n   ✗ 生成失败: {type(e).__name__}: {str(e)}")
    import traceback
    print("\n详细错误信息:")
    traceback.print_exc()

# 清理临时文件
print(f"\n6. 清理临时文件...")
try:
    os.remove(wav_path)
    print(f"   ✓ 已删除: {wav_path}")
except:
    pass

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)