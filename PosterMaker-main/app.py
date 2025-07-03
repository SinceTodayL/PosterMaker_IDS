import gradio as gr
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import datetime

import cv2
import torch
import transformers
from diffusers import FlowMatchEulerDiscreteScheduler
import textwrap

from models.adapter_models import LinearAdapterWithLayerNorm
from utils.data_processor import UserInputProcessor
from utils.sd3_utils import *
from utils.utils import post_process
from utils.data_processor import UserInputProcessor
from pipelines.pipeline_sd3 import StableDiffusion3ControlNetPipeline

def check_and_process_texts(texts_str, width, height):
    try:
        if not texts_str:
            raise ValueError("texts_str cannot be None or empty")
            
        texts = json.loads(texts_str)
        
        if not texts or not isinstance(texts, list):
            raise ValueError("Invalid texts format: must be a non-empty list")
            
        if len(texts) > 7:
            raise ValueError("Too many text lines: maximum allowed is 7 lines")
            
        processed_texts = []
        
        for text in texts:
            if not isinstance(text, dict) or 'content' not in text or 'pos' not in text:
                raise ValueError("Invalid text format: each item must be a dict with 'content' and 'pos'")
                
            content = text['content']
            pos = text['pos']
            
            # 检查文本长度
            if len(content) > 16:
                raise ValueError(f"Text too long: '{content}' exceeds 16 characters")
                
            # 检查并修正边界值
            x1, y1, x2, y2 = pos
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # 确保 x1 < x2, y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            processed_texts.append({
                "content": content,
                "pos": [x1, y1, x2, y2]
            })
            
        return processed_texts
        
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in texts_str")
    except Exception as e:
        raise ValueError(f"Error processing texts: {str(e)}")


class ImageGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_processor = UserInputProcessor()
        
        # 初始化模型和管道
        self.initialize_models()
        
        # 显示内存使用情况
        if torch.cuda.is_available():
            self.print_memory_usage()
        
    def print_memory_usage(self):
        """输出当前CUDA内存使用状态"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"CUDA内存使用状态:")
            print(f"  已分配: {allocated:.2f} GB")
            print(f"  已预留: {reserved:.2f} GB") 
            print(f"  总容量: {total:.2f} GB")
            print(f"  可用: {total - reserved:.2f} GB\n")
        
    def initialize_models(self):
        # 加载所有必要的模型组件和管道
        args = self.get_default_args()
        
        # 加载文本编码器
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
        )
        text_encoder_cls_three = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
        )
        text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
            args, text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
        ) 
        # 加载分词器
        tokenizer_one = transformers.CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )
        tokenizer_two = transformers.CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=args.revision,
        )
        tokenizer_three = transformers.T5TokenizerFast.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_3",
            revision=args.revision,
        )

        # 加载VAE模型
        vae = load_vae(args)
        # 加载SD3 Transformer模型
        transformer = load_transfomer(args)
        # 加载调度器
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        # 创建SceneGenNet控制网络
        controlnet_inpaint = load_controlnet(args, transformer, additional_in_channel=1, num_layers=23, scratch=True)
        # 创建TextRenderNet控制网络
        controlnet_text = load_controlnet(args, transformer, additional_in_channel=0, scratch=True)      
        # 加载适配器模块
        adapter = LinearAdapterWithLayerNorm(128, 4096)
        
        controlnet_inpaint.load_state_dict(torch.load(args.controlnet_model_name_or_path, map_location='cpu'))
        textrender_net_state_dict = torch.load(args.controlnet_model_name_or_path2, map_location='cpu')
        controlnet_text.load_state_dict(textrender_net_state_dict['controlnet_text'])
        adapter.load_state_dict(textrender_net_state_dict['adapter'])

        # 设置设备和数据类型
        weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = self.device

        # 将所有模型迁移至目标设备并设置精度
        vae = vae.to(device=device, dtype=weight_dtype)
        transformer = transformer.to(device=device, dtype=weight_dtype)
        text_encoder_one = text_encoder_one.to(device=device, dtype=weight_dtype)
        text_encoder_two = text_encoder_two.to(device=device, dtype=weight_dtype)
        text_encoder_three = text_encoder_three.to(device=device, dtype=weight_dtype)
        controlnet_inpaint = controlnet_inpaint.to(device=device, dtype=weight_dtype)
        controlnet_text = controlnet_text.to(device=device, dtype=weight_dtype)
        adapter = adapter.to(device=device, dtype=weight_dtype)

        # 加载推理管道
        pipeline = StableDiffusion3ControlNetPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(
                noise_scheduler.config
                ),
            vae=vae,
            transformer=transformer,
            text_encoder=text_encoder_one,
            tokenizer=tokenizer_one,
            text_encoder_2=text_encoder_two,
            tokenizer_2=tokenizer_two,
            text_encoder_3=text_encoder_three,
            tokenizer_3=tokenizer_three,
            controlnet_inpaint=controlnet_inpaint,
            controlnet_text=controlnet_text,
            adapter=adapter,
        )

        self.pipeline = pipeline.to(dtype=weight_dtype, device=device)

        # CUDA 显存优化模块开关配置
        opt_config = {
            "enable_attention_slicing": True,
            "enable_vae_slicing": True,
            "enable_cpu_offload": True,
            "enable_manual_vae_slicing": True,
            "enable_gradient_checkpointing": True,
            "enable_forward_chunking": True,
            "enable_controlnet_optimization": True,
            "enable_xformers": True,
            "enable_vae_tiling": True,
        }

        # 启用CUDA内存优化技术
        if torch.cuda.is_available():
            print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            optimizations_enabled = []

            # 注意力切片
            if opt_config["enable_attention_slicing"]:
                try:
                    if hasattr(self.pipeline, 'enable_attention_slicing'):
                        self.pipeline.enable_attention_slicing(1)
                        optimizations_enabled.append("注意力切片")
                except Exception as e:
                    print(f"无法启用注意力切片: {e}")

            # VAE切片
            if opt_config["enable_vae_slicing"]:
                try:
                    if hasattr(self.pipeline, 'enable_vae_slicing'):
                        self.pipeline.enable_vae_slicing()
                        optimizations_enabled.append("VAE切片")
                except Exception as e:
                    print(f"无法启用VAE切片: {e}")

            # CPU
            if opt_config["enable_cpu_offload"]:
                try:
                    if hasattr(self.pipeline, 'enable_sequential_cpu_offload'):
                        self.pipeline.enable_sequential_cpu_offload()
                        optimizations_enabled.append("序列CPU卸载")
                    elif hasattr(self.pipeline, 'enable_model_cpu_offload'):
                        self.pipeline.enable_model_cpu_offload()
                        optimizations_enabled.append("模型CPU卸载")
                except Exception as e:
                    print(f"无法启用CPU卸载: {e}")

            # 手动VAE切片
            if opt_config["enable_manual_vae_slicing"]:
                try:
                    if hasattr(self.pipeline, 'vae') and hasattr(self.pipeline.vae, 'enable_slicing'):
                        self.pipeline.vae.enable_slicing()
                        optimizations_enabled.append("手动VAE切片")
                except Exception as e:
                    print(f"无法启用手动VAE切片: {e}")

            # 主Transformer模块优化
            try:
                if hasattr(self.pipeline, 'transformer'):
                    if opt_config["enable_gradient_checkpointing"]:
                        if hasattr(self.pipeline.transformer, 'enable_gradient_checkpointing'):
                            self.pipeline.transformer.enable_gradient_checkpointing()
                            optimizations_enabled.append("梯度检查点")
                    if opt_config["enable_forward_chunking"]:
                        if hasattr(self.pipeline.transformer, 'enable_forward_chunking'):
                            self.pipeline.transformer.enable_forward_chunking(chunk_size=1)
                            optimizations_enabled.append("前向分块")
            except Exception as e:
                print(f"无法启用Transformer优化: {e}")

            # ControlNet 优化
            if opt_config["enable_controlnet_optimization"]:
                try:
                    for controlnet_name in ['controlnet_inpaint', 'controlnet_text']:
                        if hasattr(self.pipeline, controlnet_name):
                            controlnet = getattr(self.pipeline, controlnet_name)
                            if opt_config["enable_gradient_checkpointing"] and hasattr(controlnet, 'enable_gradient_checkpointing'):
                                controlnet.enable_gradient_checkpointing()
                                optimizations_enabled.append(f"{controlnet_name}梯度检查点")
                            if opt_config["enable_forward_chunking"] and hasattr(controlnet, 'enable_forward_chunking'):
                                controlnet.enable_forward_chunking(chunk_size=1)
                                optimizations_enabled.append(f"{controlnet_name}前向分块")
                except Exception as e:
                    print(f"无法启用ControlNet优化: {e}")

            # xformers注意力和VAE瓦片化
            try:
                if opt_config["enable_xformers"]:
                    if hasattr(self.pipeline.transformer, 'set_use_memory_efficient_attention_xformers'):
                        self.pipeline.transformer.set_use_memory_efficient_attention_xformers(True)
                        optimizations_enabled.append("xformers注意力")

                if opt_config["enable_vae_tiling"]:
                    if hasattr(self.pipeline.vae, 'enable_tiling'):
                        self.pipeline.vae.enable_tiling()
                        optimizations_enabled.append("VAE瓦片化")
            except Exception as e:
                print(f"无法启用额外优化: {e}")

            # 总结输出
            if optimizations_enabled:
                print(f"已启用内存优化技术: {', '.join(optimizations_enabled)}")
            else:
                print("无法启用任何内存优化技术（自定义管道或不兼容）")

        print("")  # 空行分隔日志

        # 输出当前模型精度（确认 float16 等是否正确设置）
        print("模型精度配置:")
        print(f"VAE数据类型: {vae.dtype}")
        print(f"Transformer数据类型: {transformer.dtype}")
        print(f"文本编码器1数据类型: {text_encoder_one.dtype}")
        print(f"文本编码器2数据类型: {text_encoder_two.dtype}")
        print(f"文本编码器3数据类型: {text_encoder_three.dtype}")
        print(f"ControlNet Inpaint数据类型: {controlnet_inpaint.dtype}")
        print(f"ControlNet Text数据类型: {controlnet_text.dtype}")
        print(f"适配器数据类型: {next(adapter.parameters()).dtype}")
        print(f"管道数据类型: {self.pipeline.dtype}\n")

        
    def generate(self, main_image, mask_image, texts_str, prompt, seed_generator):
        print("图像生成器启动")
        try:
            print("清理GPU显存...")
            # 清理显存并启用最大内存节省
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  # 更彻底的内存清理
            
            print("转换输入图像格式...")
            # 将输入图像转换为numpy格式，RGB
            main_image = np.array(main_image)
            mask = cv2.cvtColor(np.array(mask_image), cv2.COLOR_BGR2GRAY)
            print(f"主图像形状: {main_image.shape}, 掩码形状: {mask.shape}")
            
            print("解析文本布局...")
            # 解析文本布局
            texts = json.loads(texts_str)
            print(f"解析出{len(texts)}个文本元素")
            
            print("预处理输入数据...")
            # 预处理输入数据
            input_data = self.data_processor(
                image=main_image,
                mask=mask,
                texts=texts,
                prompt=prompt
            )
            print(f"数据预处理完成，输入数据键: {list(input_data.keys())}")
            
            # 执行推理前再次清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("推理前内存清理完成")
            
            # 执行推理 - 减少推理步数以节省显存
            num_steps = 28 if torch.cuda.is_available() else 1  # 进一步减少推理步数
            
            # 使用与数据预处理一致的分辨率，避免张量尺寸不匹配
            # data_processor默认使用(1024, 1024)，推理也应该使用相同分辨率
            height, width = 1024, 1024  # 与data_processor的input_size保持一致
            
            print(f"使用一致的处理分辨率: {width}x{height}")
            print(f"开始管道推理，步数: {num_steps}, 分辨率: {width}x{height}")
            
            results = self.pipeline(
                prompt=prompt,
                negative_prompt='deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW',
                height=height,
                width=width,
                control_image=[input_data['cond_image_inpaint'], input_data['controlnet_im']],
                control_mask=input_data['control_mask'],
                text_embeds=input_data['text_embeds'],
                num_inference_steps=num_steps,
                generator=seed_generator,
                controlnet_conditioning_scale=1.0,
                guidance_scale=5.0,
                num_images_per_prompt=1,
            ).images
            
            print(f"管道推理完成，生成了{len(results)}张图像")
            print(f"第一张图像类型: {type(results[0])}")
            if hasattr(results[0], 'size'):
                print(f"第一张图像尺寸: {results[0].size}")
            
            print("开始后处理...")
            # 后处理，根据im_h, im_w从rel中裁剪[0, 0, im_w, im_h]区域
            rel = post_process(results[0], input_data['target_size'])
            print(f"后处理完成，最终图像类型: {type(rel)}")
            if hasattr(rel, 'size'):
                print(f"最终图像尺寸: {rel.size}")
            
            # 保存生成的图像到results/images目录
            try:
                # 创建保存目录
                save_dir = "results/images"
                os.makedirs(save_dir, exist_ok=True)
                
                # 生成时间戳文件名
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_{timestamp}.png"
                filepath = os.path.join(save_dir, filename)
                
                # 保存图像
                if isinstance(rel, Image.Image):
                    rel.save(filepath)
                elif isinstance(rel, np.ndarray):
                    Image.fromarray(rel).save(filepath)
                
                print(f"图像已保存至: {filepath}")
            except Exception as save_e:
                print(f"保存图像时出错: {save_e}")
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print("最终显存清理完成")
            
            print("图像生成成功，返回结果")
            # 返回生成的图像
            return rel
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA内存不足: {e}")
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            return "CUDA out of memory. Try reducing image size or restart the application."
        except Exception as e:
            print(f"生成过程异常: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            # 截断错误消息以避免过长的文件名
            error_msg = str(e)
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
            return f"Error in image generation: {error_msg}"
        
    def get_default_args(self):
        # 返回默认参数配置
        class Args:
            def __init__(self):
                self.pretrained_model_name_or_path = './checkpoints/stable-diffusion-3-medium-diffusers/'
                self.controlnet_model_name_or_path='./checkpoints/ours_weights/scenegen_net-1m-0415.pth'
                self.controlnet_model_name_or_path2='./checkpoints/ours_weights/textrender_net-1m-0415.pth'
                self.revision = None
        return Args()


def visualize_layout(main_image, mask_image, texts_str, prompt,
                     font_path: str = "./assets/fonts/AlibabaPuHuiTi-3-55-Regular.ttf",
                     margin_ratio: float = 0.92):
    """
    渲染带有文字 bbox 的布局示意图  
    - texts_str -> [{'pos': (x1,y1,x2,y2), 'content': "..."}] 由外部 `check_and_process_texts` 解析
    - 文本大小、位置会根据框大小自动调整并保持居中
    """
    try:
        # -------- 1. 获取画布尺寸 -------- #
        if main_image is not None:
            height, width = main_image.shape[:2]
        elif mask_image is not None:
            height, width = mask_image.shape[:2]
        else:
            width = height = 1024

        # -------- 2. 底图与 mask 合成 -------- #
        canvas = Image.new("RGBA", (width, height), "white")

        if main_image is not None:
            pil_main = Image.fromarray(main_image).convert("RGBA")
        else:
            pil_main = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        if mask_image is not None:
            gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            alpha = Image.fromarray(np.where(gray > 127, 255, 0).astype(np.uint8))
            pil_main.putalpha(alpha)

        canvas.alpha_composite(pil_main)

        # -------- 3. 工具函数 -------- #
        def get_font(size):
            """优雅地降级到默认字体"""
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                return ImageFont.load_default()

        def optimal_font_size(text, box_w, box_h, draw):
            """二分搜索可容纳最大字号（无需传 font_path，因为 get_font 自带回退）"""
            low, high, best = 1, box_h, 1
            while low <= high:
                mid = (low + high) // 2
                font = get_font(mid)
                bbox = draw.textbbox((0, 0), text, font=font)
                txt_w = bbox[2] - bbox[0]
                txt_h = bbox[3] - bbox[1]
                if txt_w <= box_w * margin_ratio and txt_h <= box_h * margin_ratio:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            return best

        def wrap_text(text, font, box_w, draw):
            """根据宽度自动换行；返回 lines 列表"""
            words = text.split()  # 兼容英文空格，中文空格同理
            if len(words) == 1:
                # 纯中文或没有空格时：按字符粗暴截断
                wrapped = textwrap.wrap(text, width=len(text))
            else:
                wrapped = textwrap.wrap(text, width=len(words))
            # 尝试不断扩行，直到所有行宽都 <= box_w*margin_ratio
            while True:
                line_too_long = False
                for i, line in enumerate(wrapped):
                    if draw.textlength(line, font=font) > box_w * margin_ratio:
                        # 把该行再拆一半
                        midpoint = max(1, len(line) // 2)
                        wrapped[i:i+1] = [line[:midpoint], line[midpoint:]]
                        line_too_long = True
                        break
                if not line_too_long:
                    break
            return wrapped

        draw = ImageDraw.Draw(canvas)

        # -------- 4. 渲染每个文本框 -------- #
        texts = check_and_process_texts(texts_str, width, height)

        for item in texts:
            (x1, y1, x2, y2) = item["pos"]
            content = item["content"]

            box_w, box_h = x2 - x1, y2 - y1

            # 4.1 计算最佳字号
            size = optimal_font_size(content, box_w, box_h, draw)
            font = get_font(size)

            # 4.2 如果单行仍超宽则自动换行并调整字号
            lines = wrap_text(content, font, box_w, draw)
            # 若换行后总高度超框，再缩小字号
            line_height = size * 1.2
            while line_height * len(lines) > box_h * margin_ratio and size > 1:
                size -= 1
                font = get_font(size)
                lines = wrap_text(content, font, box_w, draw)
                line_height = size * 1.2

            # 4.3 计算整体文本块尺寸
            txt_h = line_height * len(lines)
            txt_w = max(draw.textlength(line, font=font) for line in lines)

            # 左上角坐标（居中）
            start_x = round(x1 + (box_w - txt_w) / 2)
            start_y = round(y1 + (box_h - txt_h) / 2)

            # 4.4 绘制 bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # 4.5 绘制文字
            for idx, line in enumerate(lines):
                draw.text((start_x, start_y + idx * line_height),
                          line, fill="blue", font=font, align="center")

        return canvas.convert("RGB")  # 与旧接口保持一致

    except Exception as e:
        # 返回 Image 以外的对象可能会破坏调用链，直接 raise 更好
        return RuntimeError(f"visualize_layout failed: {e}")


# 修改generate_image函数来使用ImageGenerator
generator = ImageGenerator()

def generate_image(main_image, mask_image, texts_str, prompt, seed):
    print("开始图像生成流程")
    print(f"输入参数 - 主图像类型: {type(main_image)}, 掩码图像类型: {type(mask_image)}")
    print(f"文本字符串长度: {len(texts_str) if texts_str else 0}, 提示词长度: {len(prompt) if prompt else 0}")
    
    if main_image is None:
        error_msg = "Error: Main image is required"
        print(f"错误: {error_msg}")
        return error_msg

    try:
        print("开始处理主图像格式...")
        # 处理main_image的格式
        if isinstance(main_image, np.ndarray):
            print(f"主图像为numpy数组，形状: {main_image.shape}")
            # 处理numpy array格式
            if main_image.ndim == 3:
                if main_image.shape[2] == 4:  # RGBA格式
                    print("检测到RGBA格式")
                    rgb_array = main_image[..., :3]
                    alpha_channel = main_image[..., 3]
                    main_image = Image.fromarray(rgb_array)
                    # 如果没有提供mask，使用alpha通道作为mask
                    if mask_image is None:
                        print("从alpha通道创建掩码")
                        mask_array = (alpha_channel > 128).astype(np.uint8) * 255
                        mask_image = Image.fromarray(mask_array)
                elif main_image.shape[2] == 3:  # RGB格式
                    print("检测到RGB格式")
                    main_image = Image.fromarray(main_image)
                    if mask_image is None:
                        error_msg = "Error: When using RGB image, a mask image must be provided"
                        print(f"错误: {error_msg}")
                        return error_msg
                else:
                    error_msg = "Error: Invalid number of channels in main image"
                    print(f"错误: {error_msg}")
                    return error_msg
            else:
                error_msg = "Error: Invalid dimensions for main image"
                print(f"错误: {error_msg}")
                return error_msg
        elif isinstance(main_image, Image.Image):
            print(f"主图像为PIL.Image，模式: {main_image.mode}，尺寸: {main_image.size}")
            # 处理PIL.Image格式
            if main_image.mode == 'RGBA':
                print("转换RGBA到RGB")
                rgb_image = main_image.convert('RGB')
                alpha_channel = main_image.split()[3]
                # 如果没有提供mask，使用alpha通道作为mask
                if mask_image is None:
                    print("从alpha通道创建掩码")
                    mask_image = alpha_channel.point(lambda x: 255 if x > 128 else 0)
                main_image = rgb_image
            elif main_image.mode != 'RGB' and mask_image is None:
                error_msg = "Error: When using RGB image, a mask image must be provided"
                print(f"错误: {error_msg}")
                return error_msg
        else:
            error_msg = "Error: Main image must be numpy array or PIL.Image format"
            print(f"错误: {error_msg}")
            return error_msg

        # 确保main_image是RGB模式
        if isinstance(main_image, Image.Image) and main_image.mode != 'RGB':
            print("转换图像到RGB模式")
            main_image = main_image.convert('RGB')

        # 处理mask_image的格式
        if mask_image is not None:
            print(f"处理掩码图像，类型: {type(mask_image)}")
            if isinstance(mask_image, np.ndarray):
                print(f"掩码图像为numpy数组，形状: {mask_image.shape}")
                mask_image = Image.fromarray(mask_image)
            elif not isinstance(mask_image, Image.Image):
                error_msg = "Error: Mask image must be numpy array or PIL.Image format"
                print(f"错误: {error_msg}")
                return error_msg
            print(f"掩码图像处理完成，尺寸: {mask_image.size}")

        print("创建随机种子生成器...")
        # 使用设定的seed
        seed_generator = torch.Generator(device=generator.device).manual_seed(int(seed))
        
        print("调用图像生成器...")
        # 使用ImageGenerator生成图像
        generated_image = generator.generate(main_image, mask_image, texts_str, prompt, seed_generator)
        
        print(f"生成器返回结果类型: {type(generated_image)}")
        
        # 检查返回的结果类型
        if isinstance(generated_image, str):
            print(f"错误: 生成过程返回错误信息: {generated_image}")
            return generated_image
        elif isinstance(generated_image, Image.Image):
            print(f"成功: 成功生成图像，尺寸: {generated_image.size}，模式: {generated_image.mode}")
            return generated_image
        elif isinstance(generated_image, np.ndarray):
            print(f"成功: 成功生成图像，numpy数组形状: {generated_image.shape}")
            return generated_image
        else:
            error_msg = f"Error: Unexpected return type from generator: {type(generated_image)}"
            print(f"错误: {error_msg}")
            return error_msg
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"异常: 生成过程中发生异常: {error_msg}")
        import traceback
        print(f"堆栈跟踪: {traceback.format_exc()}")
        return error_msg

# For debugging
# def generate_image(main_image, mask_image, texts_str, prompt, seed):
#     try:
#         # 这里是生成图像的逻辑
#         # 现在只返回一个占位图像
#         if main_image is None:
#             return "Error: Main image is required"
            
#         generated_image = Image.fromarray(main_image)  # 临时使用输入图像作为输出
#         return generated_image
    
#     except Exception as e:
#         return f"Error: {str(e)}"


# 清除所有输入和输出的函数
def clear_all():
    return [None, None, "", "", 42, None, None]

# #############
# Gradio界面部分
with gr.Blocks() as iface:
    gr.Markdown("""
    # 🎨 [CVPR2025] PosterMaker: Towards High-Quality Product Poster Generation with Accurate Text Rendering

    ## 文字海报图像生成 | A text poster image generation
                
    ## **作者 | Authors:** Yifan Gao\*, Zihang Lin\*, Chuanbin Liu, Min Zhou, Tiezheng Ge, Bo Zheng, Hongtao Xie
                                
    <div style="display: flex; gap: 10px; justify-content: left;">
        <a href="https://github.com/eafn/PosterMaker"><img src="https://img.shields.io/badge/GitHub-Repository-blue?logo=github" alt="GitHub"></a>
        <a href="https://arxiv.org/abs/2504.06632"><img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv" alt="Paper"></a>
    </div>    
    """)
    gr.Markdown("""
        ---
        ## 📝 文本布局格式 | Text Layout Format
        """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### 文本JSON格式要求 | Text JSON Format Requirements:
            ```json
            [
                {"content": "第一行文本", "pos": [x1, y1, x2, y2]},
                {"content": "第二行文本", "pos": [x1, y1, x2, y2]}
            ]
            ```
            """)
        
        with gr.Column():
            gr.Markdown("""
            ### 文本限制 | Text Limitations:
            - 最多7行文本 | Maximum 7 lines of text
            - 每行≤16个字符 | ≤16 characters per line
            - 坐标不超过图像边界 | Coordinates within image boundaries
            """)

    # 第一排：文本输入框和seed设置
    with gr.Row():
        texts_input = gr.Textbox(
            label="Input JSON text layout", 
            lines=6,
            placeholder="Enter the layout JSON here...",
            scale=1,
        )
        prompt_input = gr.Textbox(
            label="Prompt", 
            lines=6,
            placeholder="Enter the generation prompt here...",
            scale=1,
        )
        seed_input = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=10000,
            step=1,  # 步长为1，确保是整数
            value=42,
            scale=1,
        )
    
    gr.Markdown("""
        ---
        ## 📷 图像上传规则 | Image Upload Rules:
        """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### 主图像(必需) | Subject Image (Required):
            - 支持RGB格式 | Supports RGB format

            ### 蒙版图像(必需) | Mask Image (Required):
            - RGB图像必须上传额外的蒙版图像 | RGB image must have a separate mask image
            """)
        
        with gr.Column():
            gr.Markdown("""
            ### 蒙版规则 | Mask Rules:
            - 白色区域：保留的部分 | White areas: areas to keep
            - 黑色区域：生成的部分 | Black areas: areas to generate
            """)

    # 第二排：图像输入
    with gr.Row():
        with gr.Column(scale=1):
            main_image_input = gr.Image(
                label="Upload Subject Image", 
                height=400,
            )
        with gr.Column(scale=1):
            mask_image_input = gr.Image(
                label="Upload Mask Image", 
                height=400,
            )
    
    # 提醒信息
    gr.Markdown("""
        ---
        ## ⚠️ 重要提示 | Important Notes:
        """)
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### 预览步骤 | Preview Steps:
            - 请先使用"Visualize Layout"按钮预览文本布局 | Please use "Visualize Layout" button first to preview text layout
            - 确认布局无误后再点击"Generate Image"生成图像 | Click "Generate Image" after confirming the layout is correct
            """)
        
        with gr.Column():
            gr.Markdown("""
            ### 等待说明 | Wait Time:
            - 图像生成可能需要较长时间，请耐心等待 | Image generation may take some time, please be patient
            """)
        
    # 第三排：按钮
    with gr.Row():
        visualize_btn = gr.Button("Visualize Layout")
        generate_btn = gr.Button("Generate Image")
        clear_btn = gr.Button("Clear All")
    
    # 第四排：输出图像
    with gr.Row():
        with gr.Column(scale=1):
            layout_output = gr.Image(
                label="Layout Visualization", 
                height=400,
            )
        with gr.Column(scale=1):
            generated_output = gr.Image(
                label="Generated Image", 
                height=400,
            )

    gr.Markdown("""
        ---
        ## 示例 | Examples:
        """)
    # 设置示例
    examples = [
        [
            json.dumps([
                {"content": "护肤美颜贵妇乳", "pos": [69, 104, 681, 185]},
                {"content": "99.9%纯度玻色因", "pos": [165, 226, 585, 272]},
                {"content": "持久保年轻", "pos": [266, 302, 483, 347]}
            ], ensure_ascii=False),
            "The subject rests on a smooth, dark wooden table, surrounded by a few scattered leaves and delicate flowers,with a serene garden scene complete with blooming flowers and lush greenery in the background.",
            './images/rgba_images/571507774301.png',
            './images/subject_masks/571507774301.png',
            42
        ],
        [
            json.dumps([
                {"content": "增强免疫力", "pos": [38, 38, 471, 127]},
                {"content": "幼儿奶粉", "pos": [38, 143, 356, 224]},
                {"content": "易于冲调", "pos": [67, 259, 219, 296]}
            ], ensure_ascii=False),
            "The golden can of milk powder rests on a smooth, dark wooden table, surrounded by a few scattered leaves and delicate flowers, with a serene garden scene complete with blooming flowers and lush greenery in the background.",
            './images/rgba_images/652158680541.png',
            './images/subject_masks/652158680541.png',
            42
        ],
        [
            json.dumps([
                {"content": "CAB恒久气垫", "pos": [85, 101, 720, 192]},
                {"content": "持久不脱妆", "pos": [294, 226, 511, 271]}
            ], ensure_ascii=False),
            "A subject sits elegantly on smooth, light beige fabric, surrounded by a backdrop of similarly draped material that offers a silky appearance. To the left, a delicate white flower injects a subtle natural element into the composition. The overall environment is clean, bright, and minimalistic, exuding a sense of sophistication and simplicity that highlights the subject beautifully.",
            './images/rgba_images/809702153676.png',
            './images/subject_masks/809702153676.png',
            888
        ],
            [
        json.dumps([
            {"content": "原创新款", "pos": [135, 60, 686, 199]},
            {"content": "卡通款手机壳", "pos": [246, 236, 575, 299]}
        ], ensure_ascii=False),
        "The poster features a vibrant yellow background adorned with playful cartoons, including rainbows, clouds, and stars. Characters perform activities like carrying bags and holding hearts, adding a dynamic feel. Comic-style text amplifies the cheerful vibe. The solid yellow backdrop ensures the product stands out. The poster uses eye-catching fonts and text, offering clear visuals that blend harmonious design with sophistication.",
        './images/rgba_images/749870344644.png',
        './images/subject_masks/749870344644.png',
        1000
        ]
    ]
    gr.Examples(
        examples=examples,
        inputs=[texts_input, prompt_input, main_image_input, mask_image_input, seed_input]
    )

    # 包装generate_image函数，添加更好的错误处理和日志
    def generate_image_with_logging(main_image, mask_image, texts_str, prompt, seed):
        print("\n" + "="*50)
        print("图像生成流程启动")
        print("="*50)
        
        result = generate_image(main_image, mask_image, texts_str, prompt, seed)
        
        print(f"生成结果类型: {type(result)}")
        if isinstance(result, str):
            print(f"最终错误: {result}")
        else:
            print("图像生成流程完成")
        
        print("="*50)
        print("图像生成流程结束")
        print("="*50 + "\n")
        
        return result

    # 设置按钮事件
    visualize_btn.click(
        fn=visualize_layout,
        inputs=[main_image_input, mask_image_input, texts_input, prompt_input],
        outputs=layout_output
    )
    
    generate_btn.click(
        fn=generate_image_with_logging,
        inputs=[main_image_input, mask_image_input, texts_input, prompt_input, seed_input],
        outputs=generated_output
    )
    
    # 清除按钮事件
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[main_image_input, mask_image_input, texts_input, prompt_input, 
                seed_input, layout_output, generated_output]
    )


# main function
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0",server_port=7861)

