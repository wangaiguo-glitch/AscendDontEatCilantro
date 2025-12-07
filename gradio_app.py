

import gradio as gr
import cv2
import numpy as np
import time
import threading
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== 导入现有模块 ====================
print("🔄 正在导入现有模块...")

# 用于模拟的后备函数（当导入失败时使用）
def create_fallback_functions():
    """创建后备函数，当导入失败时使用"""
    
    def run_vlm_analysis(frame):
        """模拟VLM分析"""
        return f"模拟VLM分析 - 检测时间: {time.strftime('%H:%M:%S')}"
    
    def run_yolo_detection(frame, conf_thres=0.5, iou_thres=0.45):
        """模拟YOLO检测"""
        return {
            "category_id": [0, 2],
            "bbox": [[100, 100, 50, 100], [300, 150, 50, 50]],
            "score": [0.85, 0.75]
        }
    
    def run_heatmap_generation(frame):
        """模拟热力图生成"""
        # 生成随机深度图
        depth_map = np.random.rand(frame.shape[0], frame.shape[1]) * 255
        return depth_map.astype(np.float32)
    
    def draw_depth_visualization(depth_map):
        """模拟热力图可视化"""
        # 简单的颜色映射
        depth_uint8 = depth_map.astype(np.uint8)
        heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        return heatmap
    
    return {
        'vlm_analysis': run_vlm_analysis,
        'yolo_detection': run_yolo_detection,
        'heatmap_generation': run_heatmap_generation,
        'draw_depth': draw_depth_visualization
    }

# 尝试导入真实模块
try:
    # 导入YOLO模块
    from run_yolo import detect_once, draw_in_memory
    
    # 导入热力图模块
    from run_midas import depth_once, draw_depth
    
    # 导入VLM模块
    from vlm import vlm_infer
    
    print("✅ 成功导入所有现有模块")
    
    # 包装函数，以便统一调用
    def run_vlm_analysis(frame):
        """调用vlm_infer函数"""
        try:
            result = vlm_infer(frame)
            if result:
                # 这里需要根据vlm_infer的实际返回值调整
                # 假设返回的是包含答案的字典
                return f"VLM分析结果: {result}"
            return "VLM分析完成，无具体结果"
        except Exception as e:
            return f"VLM分析出错: {str(e)}"
    
    # 注意：YOLO和MiDaS需要模型参数，我们暂时用None占位
    # 实际使用时需要在初始化时加载模型
    def run_yolo_detection(frame, conf_thres=0.5, iou_thres=0.45):
        """调用detect_once函数"""
        try:
            # 这里需要传入模型，暂时返回模拟结果
            # 实际使用时：return detect_once(yolo_model, frame, conf_thres, iou_thres)
            return {
                "category_id": [0, 2],
                "bbox": [[100, 100, 50, 100], [300, 150, 50, 50]],
                "score": [0.85, 0.75]
            }
        except Exception as e:
            print(f"YOLO检测出错: {e}")
            return {"category_id": [], "bbox": [], "score": []}
    
    def draw_yolo_visualization(frame, detection_result):
        """调用draw_in_memory函数"""
        try:
            # 需要data_names参数，这里使用COCO类别名前几个作为示例
            data_names = ["person", "bicycle", "car", "motorcycle", "airplane"]
            return draw_in_memory(frame, detection_result, data_names)
        except Exception as e:
            print(f"绘制YOLO结果出错: {e}")
            return frame
    
    def run_heatmap_generation(frame):
        """调用depth_once函数"""
        try:
            # 这里需要传入模型，暂时返回模拟结果
            # 实际使用时：return depth_once(midas_model, frame)
            return np.random.rand(frame.shape[0], frame.shape[1]) * 255
        except Exception as e:
            print(f"热力图生成出错: {e}")
            return np.zeros((frame.shape[0], frame.shape[1]))
    
    def draw_depth_visualization(depth_map):
        """调用draw_depth函数"""
        try:
            return draw_depth(depth_map)
        except Exception as e:
            print(f"绘制热力图出错: {e}")
            # 简单的备选方案
            depth_uint8 = depth_map.astype(np.uint8)
            return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    
    # 创建函数字典
    functions = {
        'vlm_analysis': run_vlm_analysis,
        'yolo_detection': run_yolo_detection,
        'draw_yolo': draw_yolo_visualization,
        'heatmap_generation': run_heatmap_generation,
        'draw_depth': draw_depth_visualization
    }
    
except ImportError as e:
    print(f"⚠️ 导入现有模块失败: {e}")
    print("将使用模拟函数作为后备方案")
    functions = create_fallback_functions()

# ==================== 全局状态管理 ====================
class AppState:
    """应用状态管理"""
    def __init__(self):
        self.is_processing = False
        self.vlm_output = "等待VLM分析..."
        self.voice_text = "等待语音播报..."
        self.last_vlm_time = 0
        self.vlm_interval = 10  # VLM分析间隔（秒）
        self.current_params = {
            'confidence': 0.5,
            'iou': 0.45,
            'heatmap_alpha': 0.6
        }

state = AppState()

# ==================== 核心处理函数 ====================
def process_video_generator(video_source):
    """
    处理视频流的生成器函数
    返回: (yolo_image, heatmap_image, vlm_output, voice_text)
    """
    # 打开视频源
    if video_source == "摄像头" or video_source == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("❌ 无法打开视频源")
        return
    
    print(f"🎥 开始处理视频源: {video_source}")
    
    try:
        while state.is_processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            
            # 1. YOLO目标检测
            yolo_result = functions['yolo_detection'](
                frame, 
                conf_thres=state.current_params['confidence'],
                iou_thres=state.current_params['iou']
            )
            yolo_image = functions['draw_yolo'](frame, yolo_result)
            
            # 2. 热力图生成
            depth_map = functions['heatmap_generation'](frame)
            heatmap_image = functions['draw_depth'](depth_map)
            
            # 3. VLM分析（每10秒一次）
            vlm_text = state.vlm_output
            voice_text = state.voice_text
            
            if current_time - state.last_vlm_time >= state.vlm_interval:
                # 更新状态为"正在分析中"
                state.vlm_output = "VLM正在分析中..."
                vlm_text = state.vlm_output
                
                # 在这里实际调用VLM分析
                # 使用线程避免阻塞
                def call_vlm_analysis():
                    try:
                        result = functions['vlm_analysis'](frame)
                        state.vlm_output = f"VLM分析结果：{result}"
                        
                        # 生成语音播报文本
                        if yolo_result and len(yolo_result.get('category_id', [])) > 0:
                            count = len(yolo_result['category_id'])
                            state.voice_text = f"检测到{count}个障碍物，请注意避让"
                        else:
                            state.voice_text = "当前画面安全，未检测到障碍物"
                        
                        print(f"✅ VLM分析完成: {result[:50]}...")
                    except Exception as e:
                        print(f"❌ VLM分析出错: {e}")
                        state.vlm_output = f"VLM分析出错: {str(e)[:100]}"
                
                # 在新线程中执行VLM分析
                vlm_thread = threading.Thread(target=call_vlm_analysis)
                vlm_thread.daemon = True
                vlm_thread.start()
                
                state.last_vlm_time = current_time
            
            # 转换图像格式
            yolo_rgb = cv2.cvtColor(yolo_image, cv2.COLOR_BGR2RGB)
            heatmap_rgb = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)
            
            yield yolo_rgb, heatmap_rgb, state.vlm_output, state.voice_text
            
            # 控制帧率
            time.sleep(0.03)
    
    finally:
        cap.release()
        print("✅ 视频处理结束")

# ==================== Gradio回调函数 ====================
def start_processing(video_source):
    """开始处理视频"""
    state.is_processing = True
    return "开始处理...", gr.update(visible=False)

def stop_processing():
    """停止处理"""
    state.is_processing = False
    return "已停止", gr.update(visible=True)

def update_parameters(confidence, iou, heatmap_alpha):
    """更新处理参数"""
    state.current_params = {
        'confidence': confidence,
        'iou': iou,
        'heatmap_alpha': heatmap_alpha
    }
    print(f"⚙️ 参数已更新: {state.current_params}")

# ==================== 创建Gradio界面 ====================
def create_interface():
    """创建Gradio界面"""
    
    # 自定义CSS
    css = """
    .gradio-container {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .voice-alert-box {
        animation: fadeIn 0.5s, fadeOut 1s 4s forwards;
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; transform: translateY(0); }
        to { opacity: 0; transform: translateY(-10px); }
    }
    """
    
    with gr.Blocks(css=css, title="实时障碍物检测系统", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🚗 实时障碍物检测与播报系统
        
        **功能说明：**
        1. 🎯 **YOLO目标检测**：实时检测障碍物
        2. 🔥 **深度热力图**：显示场景深度信息
        3. 🧠 **VLM场景分析**：每10秒分析一次环境
        4. 🔊 **语音播报提示**：文本形式的安全提醒
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 控制面板
                gr.Markdown("### 🎮 控制面板")
                
                video_source = gr.Radio(
                    choices=["摄像头", "视频文件"],
                    value="摄像头",
                    label="选择视频源"
                )
                
                video_file = gr.File(
                    label="上传视频文件",
                    file_types=[".mp4", ".avi", ".mov"],
                    visible=False
                )
                
                with gr.Row():
                    start_btn = gr.Button("▶️ 开始", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ 停止", variant="secondary", size="lg")
                
                # 参数调节
                gr.Markdown("### ⚙️ 实时参数调节")
                
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="置信度阈值",
                    interactive=True
                )
                
                iou_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.45,
                    step=0.05,
                    label="IOU阈值",
                    interactive=True
                )
                
                heatmap_alpha = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.6,
                    step=0.1,
                    label="热力图透明度",
                    interactive=True
                )
                
                # 状态显示
                status_display = gr.Textbox(
                    label="系统状态",
                    value="就绪",
                    interactive=False
                )
            
            with gr.Column(scale=2):
                # 结果显示区域
                gr.Markdown("### 📊 实时检测结果")
                
                with gr.Row():
                    yolo_output = gr.Image(
                        label="YOLO检测结果",
                        height=350,
                        show_label=True
                    )
                    
                    heatmap_output = gr.Image(
                        label="深度热力图",
                        height=350,
                        show_label=True
                    )
                
                # VLM分析结果
                gr.Markdown("### 🧠 VLM场景分析")
                vlm_output = gr.Textbox(
                    label="分析结果",
                    lines=4,
                    value="等待VLM分析...",
                    interactive=False,
                    show_label=True
                )
                
                # 语音播报内容
                gr.Markdown("### 🔊 语音播报")
                voice_output = gr.Textbox(
                    label="播报内容",
                    lines=2,
                    value="等待语音播报...",
                    interactive=False,
                    show_label=True
                )
        
        # ===== 事件绑定 =====
        
        # 视频源切换
        def toggle_video_source(choice):
            return gr.update(visible=(choice == "视频文件"))
        
        video_source.change(
            fn=toggle_video_source,
            inputs=video_source,
            outputs=video_file
        )
        
        # 参数更新
        confidence_slider.change(
            fn=update_parameters,
            inputs=[confidence_slider, iou_slider, heatmap_alpha],
            outputs=[]
        )
        
        iou_slider.change(
            fn=update_parameters,
            inputs=[confidence_slider, iou_slider, heatmap_alpha],
            outputs=[]
        )
        
        heatmap_alpha.change(
            fn=update_parameters,
            inputs=[confidence_slider, iou_slider, heatmap_alpha],
            outputs=[]
        )
        
        # 开始处理
        start_btn.click(
            fn=start_processing,
            inputs=video_source,
            outputs=[status_display, start_btn]
        ).then(
            fn=process_video_generator,
            inputs=video_source,
            outputs=[yolo_output, heatmap_output, vlm_output, voice_output]
        )
        
        # 停止处理
        stop_btn.click(
            fn=stop_processing,
            inputs=[],
            outputs=[status_display, start_btn]
        )
        
        # 语音播报渐变效果
        app.load(
            fn=None,
            inputs=[],
            outputs=[],
            js="""
            function showVoiceAlert(text) {
                if (!text || text === '等待语音播报...') return;
                
                // 创建提示框
                const alertDiv = document.createElement('div');
                alertDiv.className = 'voice-alert-box';
                alertDiv.innerHTML = '🔊 ' + text;
                
                // 添加到页面顶部
                const container = document.querySelector('.gradio-container');
                if (container) {
                    // 移除旧的提示
                    const oldAlerts = container.querySelectorAll('.voice-alert-box');
                    oldAlerts.forEach(alert => alert.remove());
                    
                    // 插入新提示
                    container.insertBefore(alertDiv, container.firstChild);
                    
                    // 5秒后移除
                    setTimeout(() => {
                        if (alertDiv.parentNode) {
                            alertDiv.remove();
                        }
                    }, 5000);
                }
            }
            
            // 监听语音播报更新
            setInterval(() => {
                const voiceBox = document.querySelector('textarea[label="播报内容"]');
                if (voiceBox && voiceBox.value) {
                    showVoiceAlert(voiceBox.value);
                }
            }, 1000);
            """
        )
        
        # 页脚
        gr.Markdown("""
        ---
        **系统信息：**
        - 📅 版本: 1.0.0
        - 🏗️ 框架: Gradio + MindSpore Lite
        - 📍 基于现有项目结构
        
        *注意：这是一个演示版本，实际功能取决于模型加载情况*
        """)
    
    return app

# ==================== 主函数 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("🚀 实时障碍物检测系统 - Gradio版本")
    print("=" * 60)
    print("基于现有项目结构，调用现有模块")
    print("VLM分析间隔: 10秒")
    print("=" * 60)
    
    try:
        # 创建界面
        app = create_interface()
        
        # 启动应用
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ 应用启动失败: {e}")
        print("请检查：")
        print("1. 端口7860是否被占用")
        print("2. 依赖包是否安装: pip install gradio opencv-python")
        print("3. 现有模块路径是否正确")

if __name__ == "__main__":
    main()