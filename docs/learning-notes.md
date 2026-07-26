# Depth Anything V2 学习笔记

## 它做什么

从单张 RGB 图像或视频帧中预测相对深度图。

## 对多镜头视频的价值

- 在单个镜头内区分前景、中景、背景
- 为人物和物体提供前后、遮挡、远近的空间证据
- 不直接比较不同镜头之间的绝对距离
- 需要结合镜头切分、实体追踪和时序信息

## 我在探索 run_video.py
导入一些需要用的包，
- 'argparse' 是for命令行输入参数的解析器：
- 'glob' 用通配符匹配文件路径:
- 'cv2' 读写模型：
- ‘torch' 运行模型的框架
```python
import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2
```

if __name__ == '__main__': 一个判断，只有在运行这个文件时才会执行后面的代码，如果是别的地方import的话，name就不是'__main__'了，就不会误触。
```python

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
```
torch.load() 从磁盘上读出之前保存的模型参数文件，将文件从硬盘加载的内存里。返回一个字典，以cpu的方式加载进来，cpu的方式最稳定安全。
```python
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder]) #建了一个空模型结构，参数是随机的，还没训练。**model_congifs[] 将字典变为命名参数 a：b 变为a = b
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu')) #把参数放入空的模型结构里
    depth_anything = depth_anything.to(DEVICE).eval()
```
做判断：输入是文件还是文件夹，然后收集要处理的所有文件路径
- 如果是文件，且是.txt文件，那就逐行读取位置。读出来变为一个列表。
- splitlines()是换行切，丢掉换行符。
- 如果不是文件，那就当目录处理，glob = 批量匹配文件路径：glob.glob() 递归找
```python
    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
```
```python
    
    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        
        if args.pred_only: 
            output_width = frame_width
        else: 
            output_width = frame_width * 2 + margin_width
        
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            depth = depth_anything.infer_image(raw_frame, args.input_size)
            
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            
            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            if args.pred_only:
                out.write(depth)
            else:
                split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                combined_frame = cv2.hconcat([raw_frame, split_region, depth])
                
                out.write(combined_frame)
        
        raw_video.release()
        out.release()
```
