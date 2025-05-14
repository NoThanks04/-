import argparse
import gc
import os
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# 设置CUDA内存分割大小，避免内存碎片化
torch.backends.cuda.max_split_size_mb = 1024
# 清理GPU内存
torch.cuda.empty_cache()
gc.collect()

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='models', help='模型路径')
parser.add_argument('--screenspot_imgs', type=str, default='data/initial/screenspotv2_image', help='图像所在路径')
parser.add_argument('--screenspot_test', type=str, default='data/initial/screenspotv2_web_ug_target.json', help='测试数据JSON文件')
parser.add_argument('--output_path', type=str, default='outputs/target', help='输出路径')
parser.add_argument('--max_pixels', type=int, default=256, help='最大像素数')
parser.add_argument('--batch_size', type=int, default=1, help='批大小')
parser.add_argument('--epsilon', type=float, default=16, help='PGD扰动大小')
parser.add_argument('--iters', type=int, default=50, help='迭代次数')
parser.add_argument('--step_size', type=float, default=1, help='步长')
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_path, exist_ok=True)

print(f"加载模型: {args.model_path}")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model_path, 
    device_map="auto", 
    torch_dtype=torch.float16
).eval()

# 固定参数，不进行梯度更新
for name, param in model.named_parameters():
    param.requires_grad = False

processor = AutoProcessor.from_pretrained(args.model_path)

# 清理GPU内存
torch.cuda.empty_cache()
gc.collect()

print(f"准备处理数据: {args.screenspot_test}")

import json
from PIL import Image
import torch
import numpy as np

# 加载数据集
with open(args.screenspot_test, 'r') as f:
    dataset = json.load(f)

print(f"数据集大小: {len(dataset)}")

# 图像预处理函数
def preprocess_image(image_path):
    image = Image.open(image_path)
    image_tensor = processor.image_processor(image, return_tensors="pt").pixel_values
    return image_tensor.to(model.device), image

# 测试图像和指令
def prepare_prompts(image, instruction):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"""
Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.
Description: {instruction}
Answer:"""},
            ],
        },
    ]
    return messages

scaling_tensor = torch.tensor((0.26862954, 0.26130258, 0.27577711)).to(model.device)
scaling_tensor = scaling_tensor.reshape((3, 1, 1))
epsilon = args.epsilon / 255.0 / scaling_tensor
step_size = args.step_size / 255.0 / scaling_tensor
inverse_normalize = torchvision.transforms.Normalize(
    mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711], 
    std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711]
)

# 进行PGD攻击
def pgd_attack(model, processor, image_path, instruction, epsilon, step_size, num_iters):
    # 获取原始图像
    image = Image.open(image_path)
    
    # 转换为PyTorch张量
    image_tensor = processor.image_processor(image, return_tensors="pt").pixel_values
    image_tensor = image_tensor.to(model.device)
    
    # 初始化扰动
    delta = torch.zeros_like(image_tensor, requires_grad=True, device=model.device)
    
    # 准备输入
    messages = prepare_prompts(image, instruction)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 目标是使模型预测向左上角(0,0)的点
    target_instruction = "点击屏幕左上角(0,0)"
    target_messages = prepare_prompts(image, target_instruction)
    target_text = processor.apply_chat_template(target_messages, tokenize=False, add_generation_prompt=True)
    
    iter_bar = tqdm(range(num_iters), desc="PGD攻击")
    for i in iter_bar:
        # 添加扰动到原始图像
        adv_image_tensor = image_tensor + delta
        
        # 规范化图像值
        adv_image_tensor = torch.clamp(adv_image_tensor, 0, 1)
        
        # 处理输入
        inputs = processor(
            text=[text],
            images=[adv_image_tensor.squeeze(0)],
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        # 前向传播
        with torch.enable_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
        
        # 反向传播
        loss.backward()
        
        # 更新扰动
        delta_grad = delta.grad.detach()
        delta.data = delta.data - step_size * torch.sign(delta_grad)
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.grad.zero_()
        
        # 每10次迭代清理缓存
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        iter_bar.set_postfix(loss=loss.item())
    
    # 最终对抗图像
    adv_image_tensor = torch.clamp(image_tensor + delta, 0, 1)
    
    # 转换为PIL图像
    adv_image = tensor_to_pil(adv_image_tensor.squeeze(0))
    
    return adv_image

# 将张量转换为PIL图像
def tensor_to_pil(tensor):
    tensor = tensor.detach().cpu().numpy()
    tensor = tensor.transpose(1, 2, 0)
    tensor = (tensor * 255).astype('uint8')
    image = Image.fromarray(tensor)
    return image

# 开始攻击
print("开始PGD攻击")
for i, item in enumerate(tqdm(dataset[:5], desc="处理样本")):  # 先处理前5个样本进行测试
    try:
        img_filename = item["image"]
        img_id = item["id"]
        instruction = item["instruction"]
        
        # 图像路径
        img_path = os.path.join(args.screenspot_imgs, img_filename)
        if not os.path.exists(img_path):
            print(f"图像不存在: {img_path}")
            continue
            
        print(f"处理图像: {img_filename}, ID: {img_id}")
        print(f"指令: {instruction}")
        
        # 进行PGD攻击
        adv_image = pgd_attack(
            model, 
            processor, 
            img_path, 
            instruction, 
            epsilon, 
            step_size, 
            args.iters
        )
        
        # 保存对抗样本
        output_filename = f"{img_id}_{img_filename}"
        save_path = os.path.join(args.output_path, output_filename)
        adv_image.save(save_path)
        print(f"保存对抗样本: {save_path}")
        
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"处理样本时出错: {e}")
        continue

print("攻击完成!") 