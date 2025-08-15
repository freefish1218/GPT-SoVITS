#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冥想音频生成专业应用
Professional Meditation Audio Generation App
"""

import os
import sys
import json
import warnings
import logging
import gradio as gr
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

# 设置日志级别
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# 设置环境变量以强制使用v2ProPlus模型
os.environ["version"] = "v2ProPlus"
os.environ["is_half"] = "True"

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
try:
    # 先导入配置
    from config import get_weights_names, name2gpt_path, name2sovits_path
    
    # 获取可用的模型列表
    SoVITS_names, GPT_names = get_weights_names()
    
    # 导入推理相关函数
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
    print("请确保在 GPT-SoVITS 项目根目录下运行此脚本")
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
        "tone_description": "声音温柔、平静、充满关怀"
    },
    "深度放松": {
        "description": "适合深度放松和压力释放",
        "speed": 0.85,
        "top_k": 20,
        "top_p": 0.6,
        "temperature": 0.6,
        "pause_second": 0.5,
        "tone_description": "声音缓慢、深沉、令人安心"
    },
    "正念觉察": {
        "description": "适合正念冥想和觉察练习",
        "speed": 1.0,
        "top_k": 10,
        "top_p": 0.8,
        "temperature": 0.8,
        "pause_second": 0.35,
        "tone_description": "声音清晰、专注、富有引导性"
    },
    "睡眠引导": {
        "description": "适合睡前冥想和助眠",
        "speed": 0.8,
        "top_k": 25,
        "top_p": 0.5,
        "temperature": 0.5,
        "pause_second": 0.6,
        "tone_description": "声音轻柔、缓慢、催眠般的节奏"
    },
    "能量激活": {
        "description": "适合晨间冥想和能量提升",
        "speed": 1.1,
        "top_k": 12,
        "top_p": 0.85,
        "temperature": 0.9,
        "pause_second": 0.3,
        "tone_description": "声音充满活力、积极、振奋人心"
    },
    "自定义": {
        "description": "手动调整所有参数",
        "speed": 1.0,
        "top_k": 15,
        "top_p": 0.75,
        "temperature": 0.75,
        "pause_second": 0.35,
        "tone_description": "根据需要自行调整"
    }
}

# 专业的CSS样式
CUSTOM_CSS = """
/* 整体主题 */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
}

/* 主容器样式 */
.main-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    padding: 30px;
    margin: 20px auto;
    max-width: 1200px;
}

/* 标题样式 */
.app-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 10px;
    letter-spacing: -1px;
}

.app-subtitle {
    color: #6b7280;
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
    font-weight: 400;
}

/* 预设卡片样式 */
.preset-card {
    background: linear-gradient(145deg, #f3f4f6, #ffffff);
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 15px;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.preset-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    border-color: #667eea;
}

.preset-card.selected {
    background: linear-gradient(145deg, #ede9fe, #f3f0ff);
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.preset-title {
    font-size: 18px;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 8px;
}

.preset-description {
    font-size: 14px;
    color: #6b7280;
    line-height: 1.5;
}

/* 参数区域样式 */
.params-section {
    background: #f9fafb;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
}

.params-title {
    font-size: 16px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* 滑块样式 */
.gr-slider {
    margin-bottom: 20px;
}

.gr-slider .gr-slider-container {
    background: #e5e7eb;
    height: 6px;
    border-radius: 3px;
}

.gr-slider .gr-slider-fill {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 3px;
}

.gr-slider .gr-slider-thumb {
    width: 20px;
    height: 20px;
    background: white;
    border: 3px solid #667eea;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* 按钮样式 */
.generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    padding: 15px 40px !important;
    border-radius: 12px !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
}

.generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}

.secondary-btn {
    background: white !important;
    color: #667eea !important;
    border: 2px solid #667eea !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.secondary-btn:hover {
    background: #f3f0ff !important;
}

/* 音频播放器样式 */
.gr-audio {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* 文本输入框样式 */
.gr-text-input, .gr-textbox {
    border-radius: 8px !important;
    border: 2px solid #e5e7eb !important;
    transition: all 0.3s ease !important;
    font-size: 15px !important;
}

.gr-text-input:focus, .gr-textbox:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* 下拉框样式 */
.gr-dropdown {
    border-radius: 8px !important;
    border: 2px solid #e5e7eb !important;
}

/* 分组样式 */
.gr-group {
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
    background: white !important;
    padding: 20px !important;
}

/* 标签样式 */
.gr-label {
    color: #374151 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    margin-bottom: 8px !important;
}

/* 提示信息样式 */
.info-box {
    background: linear-gradient(145deg, #f0f9ff, #e0f2fe);
    border-left: 4px solid #3b82f6;
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
}

.info-box-title {
    font-weight: 600;
    color: #1e40af;
    margin-bottom: 5px;
}

.info-box-content {
    color: #3730a3;
    font-size: 14px;
}

/* 加载动画 */
@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.5;
    }
}

.loading {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .app-title {
        font-size: 32px;
    }
    
    .main-container {
        padding: 20px;
        margin: 10px;
    }
}
"""

# JavaScript 代码
CUSTOM_JS = """
function updatePresetSelection(presetName) {
    // 更新预设卡片的选中状态
    document.querySelectorAll('.preset-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    const selectedCard = document.querySelector(`[data-preset="${presetName}"]`);
    if (selectedCard) {
        selectedCard.classList.add('selected');
    }
}

// 平滑滚动
function smoothScroll(target) {
    document.querySelector(target).scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// 初始化工具提示
function initTooltips() {
    // 实现工具提示功能
}
"""

class MeditationApp:
    """冥想音频生成应用类"""
    
    def __init__(self):
        self.current_preset = "平静舒缓"
        self.load_models()
    
    def load_models(self):
        """加载模型配置"""
        # 检查是否有可用的模型
        if not SoVITS_names or not GPT_names:
            print("警告：未找到可用的模型文件")
            return
        
        # 选择最新的模型
        self.current_sovits = SoVITS_names[0] if SoVITS_names else None
        self.current_gpt = GPT_names[-1] if GPT_names else None
    
    def apply_preset(self, preset_name: str) -> Dict:
        """应用预设参数"""
        self.current_preset = preset_name
        preset = MEDITATION_PRESETS[preset_name]
        
        return {
            "speed": preset["speed"],
            "top_k": preset["top_k"],
            "top_p": preset["top_p"],
            "temperature": preset["temperature"],
            "pause_second": preset["pause_second"],
            "tone_desc": preset["tone_description"]
        }
    
    def generate_audio(
        self,
        text: str,
        ref_audio,
        ref_text: str,
        preset_name: str,
        speed: float,
        top_k: int,
        top_p: float,
        temperature: float,
        pause_second: float,
        text_language: str,
        ref_language: str,
        how_to_cut: str,
        sovits_model: str,
        gpt_model: str,
        progress=None
    ):
        """生成冥想音频"""
        
        print("\n" + "="*60)
        print("=== 开始生成冥想音频 ===")
        print("="*60)
        print(f"[DEBUG] 函数被调用，所有参数:")
        print(f"  - text长度: {len(text) if text else 0}")
        print(f"  - text内容: {text[:100] if text else 'None'}...")
        print(f"  - ref_audio: {ref_audio}")
        print(f"  - ref_audio类型: {type(ref_audio)}")
        print(f"  - ref_text: {ref_text}")
        print(f"  - preset_name: {preset_name}")
        print(f"  - speed: {speed}")
        print(f"  - top_k: {top_k}")
        print(f"  - top_p: {top_p}")
        print(f"  - temperature: {temperature}")
        print(f"  - pause_second: {pause_second}")
        print(f"  - text_language: {text_language}")
        print(f"  - ref_language: {ref_language}")
        print(f"  - how_to_cut: {how_to_cut}")
        print(f"  - sovits_model: {sovits_model}")
        print(f"  - gpt_model: {gpt_model}")
        print("="*60)
        
        # 参数验证
        if not text:
            print("[ERROR] 未输入文本")
            gr.Warning("请输入冥想引导文本")
            return None
        
        if not ref_audio:
            print("[ERROR] 未上传参考音频")
            gr.Warning("请上传参考音频")
            return None
        
        if not ref_text:
            print("[ERROR] 未输入参考文本")
            gr.Warning("请输入参考音频的文本")
            return None
        
        # 检查音频文件
        import os
        if isinstance(ref_audio, str):
            if not os.path.exists(ref_audio):
                print(f"[ERROR] 音频文件不存在: {ref_audio}")
                gr.Error(f"音频文件不存在: {ref_audio}")
                return None
            print(f"[DEBUG] 音频文件存在: {ref_audio}")
            print(f"[DEBUG] 音频文件大小: {os.path.getsize(ref_audio)} bytes")
            
            # 如果是MP3，尝试转换为WAV
            if ref_audio.lower().endswith('.mp3'):
                print("[INFO] 检测到MP3格式，正在转换为WAV...")
                try:
                    import librosa
                    import soundfile as sf
                    import tempfile
                    
                    # 读取MP3
                    audio_data, sr = librosa.load(ref_audio, sr=None)
                    
                    # 创建临时WAV文件
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                        wav_path = tmp_wav.name
                        sf.write(wav_path, audio_data, sr)
                        print(f"[INFO] MP3已转换为WAV: {wav_path}")
                        ref_audio = wav_path
                except Exception as e:
                    print(f"[ERROR] MP3转换失败: {e}")
                    gr.Error(f"音频格式转换失败: {e}")
                    return None
        
        try:
            # 显示进度
            if progress:
                progress(0.1, desc="正在加载模型...")
            
            # 更新模型（如果需要）
            if sovits_model and sovits_model != self.current_sovits:
                print(f"切换SoVITS模型: {self.current_sovits} -> {sovits_model}")
                change_sovits_weights(sovits_model)
                self.current_sovits = sovits_model
            
            if gpt_model and gpt_model != self.current_gpt:
                print(f"切换GPT模型: {self.current_gpt} -> {gpt_model}")
                change_gpt_weights(gpt_model)
                self.current_gpt = gpt_model
            
            if progress:
                progress(0.3, desc="正在生成音频...")
            print("[DEBUG] 开始调用get_tts_wav函数...")
            print(f"[DEBUG] 调用参数:")
            print(f"  - ref_wav_path: {ref_audio}")
            print(f"  - prompt_text: {ref_text}")
            print(f"  - prompt_language: {ref_language}")
            print(f"  - text: {text[:50]}...")
            print(f"  - text_language: {text_language}")
            
            # 生成音频 - get_tts_wav 返回一个生成器
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
            
            print(f"[DEBUG] get_tts_wav返回结果类型: {type(result_generator)}")
            
            if result_generator is None:
                print("[ERROR] get_tts_wav返回None")
                gr.Error("音频生成失败：返回空结果")
                return None
            
            # 处理生成器返回的音频
            audio_segments = []
            segment_count = 0
            
            # 计算文本切分数量来估算进度
            import re
            if "按中文句号。切" in how_to_cut:
                estimated_segments = len(re.split('[。！？]', text))
            elif "按英文句号.切" in how_to_cut:
                estimated_segments = len(re.split('[.!?]', text))
            else:
                estimated_segments = 1
            
            # 迭代生成器获取音频
            print("[DEBUG] 开始迭代生成器...")
            try:
                for sr, audio in result_generator:
                    segment_count += 1
                    if progress:
                        progress_val = 0.3 + (0.6 * segment_count / max(estimated_segments, segment_count))
                        progress(progress_val, desc=f"正在生成音频片段 {segment_count}/{estimated_segments}...")
                    
                    print(f"[DEBUG] 生成音频片段 #{segment_count}: 采样率={sr}, 音频形状={audio.shape if hasattr(audio, 'shape') else 'N/A'}")
                    audio_segments.append((sr, audio))
            except Exception as e:
                print(f"[ERROR] 迭代生成器时出错: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                gr.Error(f"处理音频时出错: {str(e)}")
                return None
            
            if audio_segments:
                if progress:
                    progress(0.9, desc="正在合并音频...")
                
                # 如果只有一个片段，直接返回
                if len(audio_segments) == 1:
                    sr, audio = audio_segments[0]
                    print(f"[SUCCESS] 生成单个音频成功: 采样率={sr}, 形状={audio.shape}")
                    if progress:
                        progress(1.0, desc="生成完成！")
                    return (sr, audio)
                
                # 如果有多个片段，合并它们
                import numpy as np
                sr = audio_segments[0][0]  # 使用第一个片段的采样率
                all_audio = []
                
                for seg_sr, seg_audio in audio_segments:
                    if seg_sr != sr:
                        print(f"警告: 采样率不一致 {seg_sr} != {sr}")
                    all_audio.append(seg_audio)
                
                # 合并所有音频片段
                merged_audio = np.concatenate(all_audio)
                print(f"[SUCCESS] 合并 {len(audio_segments)} 个音频片段，总长度: {merged_audio.shape}")
                
                if progress:
                    progress(1.0, desc="生成完成！")
                return (sr, merged_audio)
            else:
                print("get_tts_wav未返回任何音频数据")
                gr.Warning("音频生成失败：未返回音频数据")
                return None
                
        except Exception as e:
            print(f"[CRITICAL ERROR] 音频生成异常: {type(e).__name__}: {str(e)}")
            import traceback
            print("[TRACEBACK]:")
            traceback.print_exc()
            print("="*60)
            gr.Error(f"音频生成失败：{str(e)}")
            return None
    
    def create_interface(self):
        """创建Gradio界面"""
        
        with gr.Blocks(
            title="冥想音频生成专业版",
            css=CUSTOM_CSS,
            js=CUSTOM_JS,
            theme=gr.themes.Soft(
                primary_hue="purple",
                secondary_hue="indigo",
                neutral_hue="gray",
                font=gr.themes.GoogleFont("Inter")
            )
        ) as app:
            
            with gr.Column(elem_classes="main-container"):
                # 标题区域
                gr.HTML("""
                    <h1 class="app-title">冥想音频生成专业版</h1>
                    <p class="app-subtitle">使用先进的AI技术，为您创造完美的冥想引导声音</p>
                """)
                
                # 主要内容区域
                with gr.Row():
                    # 左侧：参数控制面板
                    with gr.Column(scale=1):
                        gr.Markdown("## 🎯 快速预设")
                        
                        # 预设选择
                        preset_selector = gr.Radio(
                            choices=list(MEDITATION_PRESETS.keys()),
                            value="平静舒缓",
                            label="选择冥想场景",
                            interactive=True,
                            elem_classes="preset-selector"
                        )
                        
                        # 预设描述
                        preset_info = gr.Markdown(
                            value=f"**{MEDITATION_PRESETS['平静舒缓']['description']}**\n\n"
                                  f"*{MEDITATION_PRESETS['平静舒缓']['tone_description']}*"
                        )
                        
                        # 高级参数
                        with gr.Accordion("⚙️ 高级参数设置", open=False):
                            speed_slider = gr.Slider(
                                minimum=0.6,
                                maximum=1.5,
                                value=0.95,
                                step=0.05,
                                label="语速",
                                info="调整说话速度（越小越慢）"
                            )
                            
                            pause_slider = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.4,
                                step=0.05,
                                label="句间停顿（秒）",
                                info="句子之间的停顿时长"
                            )
                            
                            with gr.Group():
                                gr.Markdown("### 🧠 AI生成参数")
                                
                                top_k_slider = gr.Slider(
                                    minimum=1,
                                    maximum=50,
                                    value=15,
                                    step=1,
                                    label="Top-K",
                                    info="控制生成的多样性"
                                )
                                
                                top_p_slider = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.7,
                                    step=0.05,
                                    label="Top-P",
                                    info="控制生成的随机性"
                                )
                                
                                temperature_slider = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.7,
                                    step=0.05,
                                    label="Temperature",
                                    info="控制生成的创造性"
                                )
                        
                        # 模型选择
                        with gr.Accordion("🤖 模型选择", open=False):
                            sovits_dropdown = gr.Dropdown(
                                choices=SoVITS_names,
                                value=SoVITS_names[0] if SoVITS_names else None,
                                label="SoVITS 模型",
                                interactive=True
                            )
                            
                            gpt_dropdown = gr.Dropdown(
                                choices=GPT_names,
                                value=GPT_names[-1] if GPT_names else None,
                                label="GPT 模型",
                                interactive=True
                            )
                    
                    # 右侧：输入和输出区域
                    with gr.Column(scale=2):
                        # 参考音频设置
                        with gr.Group():
                            gr.Markdown("### 🎤 参考音频设置")
                            
                            with gr.Row():
                                ref_audio_input = gr.Audio(
                                    label="上传参考音频（3-10秒）",
                                    type="filepath",
                                    elem_classes="audio-input"
                                )
                                
                                with gr.Column():
                                    ref_text_input = gr.Textbox(
                                        label="参考音频文本",
                                        placeholder="请输入参考音频中说的内容...",
                                        lines=3,
                                        max_lines=5
                                    )
                                    
                                    ref_language = gr.Dropdown(
                                        choices=list(dict_language.keys()),
                                        value=i18n("中文"),
                                        label="参考音频语言"
                                    )
                        
                        # 冥想文本输入
                        with gr.Group():
                            gr.Markdown("### ✍️ 冥想引导文本")
                            
                            text_input = gr.Textbox(
                                label="输入冥想引导词",
                                placeholder="请输入您想要生成的冥想引导文本...\n\n例如：\n现在，让我们开始今天的冥想练习。\n请找一个舒适的姿势坐下，轻轻闭上眼睛。\n深深地吸一口气，然后缓缓地呼出...",
                                lines=10,
                                max_lines=20
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
                        generate_btn = gr.Button(
                            "🎵 生成冥想音频",
                            variant="primary",
                            size="lg",
                            elem_classes="generate-btn"
                        )
                        
                        # 输出音频
                        audio_output = gr.Audio(
                            label="生成的冥想音频",
                            type="numpy",
                            elem_classes="audio-output"
                        )
                        
                        # 提示信息
                        gr.HTML("""
                            <div class="info-box">
                                <div class="info-box-title">💡 使用提示</div>
                                <div class="info-box-content">
                                    • 参考音频决定生成音频的音色和风格<br>
                                    • 选择合适的预设可以快速获得理想效果<br>
                                    • 文本越长，生成时间越长，建议分段生成<br>
                                    • v2ProPlus模型支持更自然的语音合成
                                </div>
                            </div>
                        """)
                
                # 事件处理
                def update_preset_info(preset_name):
                    """更新预设信息显示"""
                    preset = MEDITATION_PRESETS[preset_name]
                    info_text = f"**{preset['description']}**\n\n*{preset['tone_description']}*"
                    
                    return (
                        info_text,
                        preset["speed"],
                        preset["pause_second"],
                        preset["top_k"],
                        preset["top_p"],
                        preset["temperature"]
                    )
                
                # 预设选择事件
                preset_selector.change(
                    fn=update_preset_info,
                    inputs=[preset_selector],
                    outputs=[
                        preset_info,
                        speed_slider,
                        pause_slider,
                        top_k_slider,
                        top_p_slider,
                        temperature_slider
                    ]
                )
                
                # 生成按钮事件 - 添加调试信息
                def generate_wrapper(*args, progress=gr.Progress()):
                    """包装函数，用于调试"""
                    print("\n" + "="*60)
                    print("[BUTTON CLICK] 生成按钮被点击！")
                    print(f"[BUTTON CLICK] 收到 {len(args)} 个参数")
                    for i, arg in enumerate(args):
                        if isinstance(arg, str) and len(arg) > 100:
                            print(f"  参数 {i}: {type(arg).__name__} = {arg[:100]}...")
                        else:
                            print(f"  参数 {i}: {type(arg).__name__} = {arg}")
                    print("="*60)
                    
                    try:
                        # 调用实际的生成函数，传递progress参数
                        result = self.generate_audio(*args, progress=progress)
                        print(f"[BUTTON CLICK] 生成完成，返回类型: {type(result)}")
                        return result
                    except Exception as e:
                        print(f"[BUTTON CLICK ERROR] {type(e).__name__}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        raise
                
                generate_btn.click(
                    fn=generate_wrapper,
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
                    outputs=[audio_output],
                    show_progress=True
                )
            
            return app

def main():
    """主函数"""
    # 创建应用实例
    app = MeditationApp()
    
    # 创建界面
    interface = app.create_interface()
    
    # 启动应用
    interface.launch(
        server_name="0.0.0.0",
        server_port=9873,
        inbrowser=True,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()