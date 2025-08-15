#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冥想音频生成专业应用 - 修复版
"""

import os
import sys
import json
import warnings
import logging
import gradio as gr
import torch
import numpy as np

# 设置日志级别
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# 设置环境变量
os.environ["version"] = "v2ProPlus"
os.environ["is_half"] = "True"

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
try:
    from config import get_weights_names, name2gpt_path, name2sovits_path
    SoVITS_names, GPT_names = get_weights_names()
    
    from GPT_SoVITS.inference_webui import (
        get_tts_wav,
        change_sovits_weights,
        change_gpt_weights,
        device,
        is_half,
        i18n,
        dict_language
    )
except ImportError as e:
    print(f"导入错误：{e}")
    sys.exit(1)

# 冥想场景预设配置
MEDITATION_PRESETS = {
    "平静舒缓": {
        "description": "适合日常冥想练习，声音温和平静",
        "speed": 0.95,
        "top_k": 15,
        "top_p": 0.7,
        "temperature": 0.7,
        "pause_second": 0.4,
    },
    "深度放松": {
        "description": "适合深度放松和压力释放",
        "speed": 0.85,
        "top_k": 20,
        "top_p": 0.6,
        "temperature": 0.6,
        "pause_second": 0.5,
    },
    "正念觉察": {
        "description": "适合正念冥想和觉察练习",
        "speed": 1.0,
        "top_k": 10,
        "top_p": 0.8,
        "temperature": 0.8,
        "pause_second": 0.35,
    },
    "睡眠引导": {
        "description": "适合睡前冥想和助眠",
        "speed": 0.8,
        "top_k": 25,
        "top_p": 0.5,
        "temperature": 0.5,
        "pause_second": 0.6,
    },
}

class MeditationApp:
    """冥想音频生成应用类"""
    
    def __init__(self):
        self.current_preset = "平静舒缓"
        self.current_sovits = SoVITS_names[0] if SoVITS_names else None
        self.current_gpt = GPT_names[-1] if GPT_names else None
        print(f"初始化完成: SoVITS={self.current_sovits}, GPT={self.current_gpt}")
    
    def generate_audio_simple(
        self,
        text,
        ref_audio,
        ref_text,
        preset_name,
        speed,
        top_k,
        top_p,
        temperature,
        pause_second,
        text_language,
        ref_language,
        how_to_cut,
        sovits_model,
        gpt_model
    ):
        """简化的生成函数，不使用progress"""
        print("\n" + "="*60)
        print("生成音频被调用")
        print(f"文本: {text[:50] if text else 'None'}...")
        print(f"参考音频: {ref_audio}")
        print(f"参考文本: {ref_text}")
        print("="*60)
        
        # 参数验证
        if not text:
            return gr.Warning("请输入冥想引导文本")
        
        if not ref_audio:
            return gr.Warning("请上传参考音频")
        
        if not ref_text:
            return gr.Warning("请输入参考音频的文本")
        
        try:
            # 处理MP3文件
            if isinstance(ref_audio, str) and ref_audio.lower().endswith('.mp3'):
                print("转换MP3为WAV...")
                import librosa
                import soundfile as sf
                import tempfile
                
                audio_data, sr = librosa.load(ref_audio, sr=None)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                    wav_path = tmp_wav.name
                    sf.write(wav_path, audio_data, sr)
                    ref_audio = wav_path
                    print(f"已转换为: {wav_path}")
            
            # 更新模型
            if sovits_model and sovits_model != self.current_sovits:
                change_sovits_weights(sovits_model)
                self.current_sovits = sovits_model
            
            if gpt_model and gpt_model != self.current_gpt:
                change_gpt_weights(gpt_model)
                self.current_gpt = gpt_model
            
            print("调用 get_tts_wav...")
            
            # 生成音频
            result_generator = get_tts_wav(
                ref_wav_path=ref_audio,
                prompt_text=ref_text,
                prompt_language=ref_language,
                text=text,
                text_language=text_language,
                how_to_cut=how_to_cut,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                ref_free=False,
                speed=speed,
                if_freeze=False,
                inp_refs=None,
                sample_steps=8,
                if_sr=False,
                pause_second=pause_second
            )
            
            # 收集所有音频片段
            audio_segments = []
            for sr, audio in result_generator:
                print(f"生成片段: sr={sr}, shape={audio.shape if hasattr(audio, 'shape') else 'N/A'}")
                audio_segments.append((sr, audio))
            
            if audio_segments:
                if len(audio_segments) == 1:
                    sr, audio = audio_segments[0]
                    print(f"生成成功! 采样率={sr}")
                    return (sr, audio)
                else:
                    # 合并多个片段
                    sr = audio_segments[0][0]
                    all_audio = []
                    for seg_sr, seg_audio in audio_segments:
                        all_audio.append(seg_audio)
                    merged_audio = np.concatenate(all_audio)
                    print(f"合并 {len(audio_segments)} 个片段成功")
                    return (sr, merged_audio)
            else:
                print("未生成任何音频")
                return None
                
        except Exception as e:
            print(f"错误: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def create_interface():
    """创建Gradio界面"""
    
    app_instance = MeditationApp()
    
    with gr.Blocks(title="冥想音频生成专业版") as demo:
        
        gr.Markdown("# 🧘 冥想音频生成专业版")
        gr.Markdown("使用先进的AI技术，为您创造完美的冥想引导声音")
        
        with gr.Row():
            # 左侧控制面板
            with gr.Column(scale=1):
                gr.Markdown("## 🎯 快速预设")
                
                preset_selector = gr.Radio(
                    choices=list(MEDITATION_PRESETS.keys()),
                    value="平静舒缓",
                    label="选择冥想场景"
                )
                
                # 参数设置
                with gr.Accordion("⚙️ 高级参数", open=False):
                    speed_slider = gr.Slider(0.6, 1.5, 0.95, step=0.05, label="语速")
                    pause_slider = gr.Slider(0.1, 1.0, 0.4, step=0.05, label="句间停顿（秒）")
                    top_k_slider = gr.Slider(1, 50, 15, step=1, label="Top-K")
                    top_p_slider = gr.Slider(0.1, 1.0, 0.7, step=0.05, label="Top-P")
                    temperature_slider = gr.Slider(0.1, 1.0, 0.7, step=0.05, label="Temperature")
                
                # 模型选择
                with gr.Accordion("🤖 模型选择", open=False):
                    sovits_dropdown = gr.Dropdown(
                        choices=SoVITS_names,
                        value=app_instance.current_sovits,
                        label="SoVITS 模型"
                    )
                    gpt_dropdown = gr.Dropdown(
                        choices=GPT_names,
                        value=app_instance.current_gpt,
                        label="GPT 模型"
                    )
            
            # 右侧输入输出
            with gr.Column(scale=2):
                gr.Markdown("### 🎤 参考音频设置")
                
                with gr.Row():
                    ref_audio_input = gr.Audio(
                        label="上传参考音频（3-10秒）",
                        type="filepath"
                    )
                    
                    with gr.Column():
                        ref_text_input = gr.Textbox(
                            label="参考音频文本",
                            placeholder="输入参考音频中说的内容...",
                            lines=2
                        )
                        ref_language = gr.Dropdown(
                            choices=list(dict_language.keys()),
                            value=i18n("中文"),
                            label="参考音频语言"
                        )
                
                gr.Markdown("### ✍️ 冥想引导文本")
                
                text_input = gr.Textbox(
                    label="输入冥想引导词",
                    placeholder="请输入您想要生成的冥想引导文本...",
                    lines=5
                )
                
                with gr.Row():
                    text_language = gr.Dropdown(
                        choices=list(dict_language.keys()),
                        value=i18n("中文"),
                        label="文本语言"
                    )
                    cut_method = gr.Dropdown(
                        choices=[
                            i18n("不切"),
                            i18n("凑四句一切"),
                            i18n("凑50字一切"),
                            i18n("按中文句号。切"),
                            i18n("按英文句号.切"),
                            i18n("按标点符号切"),
                        ],
                        value=i18n("按中文句号。切"),
                        label="文本切分方式"
                    )
                
                # 生成按钮
                generate_btn = gr.Button("🎵 生成冥想音频", variant="primary", size="lg")
                
                # 输出音频
                audio_output = gr.Audio(label="生成的冥想音频", type="numpy")
        
        # 预设选择事件
        def update_preset(preset_name):
            preset = MEDITATION_PRESETS[preset_name]
            return (
                preset["speed"],
                preset["pause_second"],
                preset["top_k"],
                preset["top_p"],
                preset["temperature"]
            )
        
        preset_selector.change(
            fn=update_preset,
            inputs=[preset_selector],
            outputs=[speed_slider, pause_slider, top_k_slider, top_p_slider, temperature_slider]
        )
        
        # 生成按钮事件 - 直接绑定，不用wrapper
        generate_btn.click(
            fn=app_instance.generate_audio_simple,
            inputs=[
                text_input,
                ref_audio_input,
                ref_text_input,
                preset_selector,
                speed_slider,
                top_k_slider,
                top_p_slider,
                temperature_slider,
                pause_slider,
                text_language,
                ref_language,
                cut_method,
                sovits_dropdown,
                gpt_dropdown
            ],
            outputs=[audio_output]
        )
    
    return demo

def main():
    """主函数"""
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=9873,
        inbrowser=False,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()