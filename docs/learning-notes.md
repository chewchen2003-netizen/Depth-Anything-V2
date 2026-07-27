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

    #这两行固定就好，不需要进入循环。
    margin_width = 50
    #调用matplotlib里的颜色映射表，spectral_r这个colormaps。_r是反向，蓝绿黄红。近冷色，原暖色。
    # colormaps函数（一个颜色表仓库）可以把单通道的 depth （0，1）转成彩色 RGB（红，绿，蓝）。
    # get_cmap()在颜色表仓库中拿一个叫做spectral_r的颜色表
    #其实就是通过不同的深度映射成不同的颜色。
    cmap = matplotlib.colormaps.get_cmap('Spectral_r') 
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        #**每个**视频里的"准备工作"
        raw_video = cv2.VideoCapture(filename) #打开视频
        # .get()读视频属性的方法，返回的是float，但宽高帧率需要是整数。
        # .CAP_PROP_FRAME_WIDTH: capture property属性
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 读帧率是为了让输出视频跟原视频播放速度一致。
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        # 只输出深度预测图，不要原视频
        if args.pred_only: 
            output_width = frame_width
        else: 
            output_width = frame_width * 2 + margin_width
        # .basename() 找到最后一个/之后的内容
        # .splittext() 拆后缀，靠最后一个.点来拆
        # 这里后缀加上.mp4，也是希望输出最后都统一成mp4
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
        # VideoWriter = 视频写入器，和VideoCapture视频读取器相对应
        # fourcc = Four Character Code = 四字符编码 用来告诉 OpenCV 用什么编码格式压缩视频
        # frame_rate，输出视频的帧率，帧大小（宽高）用元组传  opencv的api设计 cv2.VideoWriter(filename, fourcc, fps, frameSize)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        
        while raw_video.isOpened():
            #ret：return value 返回值，read() 一次会返回两个值 ret = True 是否成功（布尔值），raw_frame这一帧
            # raw_frame 是一个 numpy 数组（多维数组）(H, W, 3) H：视频高度（行数） W：视频宽度（列数）3 = BGR 是哪个通道
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            #把当前帧送给 Depth Anything V2 模型，返回一张深度图。深度图的信息为离相机的远近 数组里的每一个位置 = 一个像素。每个像素存一个浮点数（深度值）。
            depth = depth_anything.infer_image(raw_frame, args.input_size)
            # min-max 归一化公式：(x - 最小值) / (最大值 - 最小值) * 255 把任意范围的数值，线性映射到 [0, 255]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            
            if args.grayscale: #np.repeat = 沿某个轴把每个元素复制 N 次
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1) #depth 当前是 (H, W)，用 np.newaxis 在最后加一个新轴：newaxis 3:沿最后一维（通道维）重复 3 次,把单通道复制成 3 份完全一样 彩色图：3 个值 = R G B
            else: #cmap(depth) 把深度映射成颜色图  matplotlib 的 colormap 函数固定输出 4 通道 RGBA。
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8) #array 形状是 (H, W, 4) cmap(depth) 输出的是 0~1 的浮点颜色。[:, :, ::-1] 通道反转 因为OpenCV 写视频用 BGR

            if args.pred_only:
                out.write(depth)
            else: #split_region 中间的白条 np.ones创造全1的数组
                split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                combined_frame = cv2.hconcat([raw_frame, split_region, depth]) # .hconcat = 横向拼接（horizontal concatenation）
                
                out.write(combined_frame)
        #让视频对象把自己占用的系统资源释放掉。
        raw_video.release() # 关闭输入视频
        out.release() # 关闭输出视频
```
