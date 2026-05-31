import cupy as cp
import cv2
import numpy as np
import os

base_path = os.path.dirname(os.path.abspath(__file__))
kernel_path = os.path.join(base_path, "ising_model.cu")
with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()

module = cp.RawModule(code=cuda_source, options=('-use_fast_math',))
isingstep = module.get_function('isingstep')
render_spin_field = module.get_function('render_spin_field')

# ========== 3. 初始化参数 ==========
# 定义画面大小 (需保证宽度是 32 的倍数)
width_pixels = 2048
height_pixels = 2048
x_words_num = width_pixels // 32
y_lines = height_pixels

# 物理参数
J = 1.0
precision = 12 # 概率掩码精度

# 初始化随机自旋场和图像缓冲区
# 随机生成无符号 32 位整数
spin_field = cp.random.randint(0, 0xFFFFFFFF, size=(y_lines, x_words_num), dtype=cp.uint32)
# 分配图像显存 (注意这里形状：高度, 宽度, 通道数4)
image_buffer = cp.zeros((y_lines, width_pixels, 4), dtype=cp.uint8)
tot_energy = cp.zeros(1, dtype=cp.float32)

# 网格与线程块配置
block_dim = (32, 8) 
# isingstep 网格大小（处理 word）
grid_ising = ((x_words_num + block_dim[0] - 1) // block_dim[0], 
              (y_lines + block_dim[1] - 1) // block_dim[1])
# render 网格大小（处理 pixel）
grid_render = ((width_pixels + block_dim[0] - 1) // block_dim[0], 
               (y_lines + block_dim[1] - 1) // block_dim[1])

# ========== 4. OpenCV UI 与主循环 ==========
cv2.namedWindow("Ising 2D Simulation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Ising 2D Simulation", 800, 800)

# 添加滑动条控制温度。真实温度 = 滑块值 / 100
# 临界温度大约为 2.269
cv2.createTrackbar("Temperature", "Ising 2D Simulation", 227, 500, lambda x: None)

frames = 0
print("Simulation Started. Press 'ESC' to exit.")

while True:
    # 获取当前滑动条温度
    trackbar_val = cv2.getTrackbarPos("Temperature", "Ising 2D Simulation")
    temperature = max(0.01, trackbar_val / 100.0) # 防止除零

    # --- 物理步 ---
    # CuPy 要求参数类型必须严格匹配 C 类型，因此需要 cp.int32, cp.float32 包装
    # 第一步：更新红格子
    isingstep(grid_ising, block_dim, (
        spin_field, tot_energy, cp.bool_(True),
        cp.int32(x_words_num), cp.int32(y_lines),
        cp.float32(temperature), cp.float32(J), cp.int32(precision), cp.int32(frames)
    ))
    
    # 第二步：更新黑格子
    isingstep(grid_ising, block_dim, (
        spin_field, tot_energy, cp.bool_(False),
        cp.int32(x_words_num), cp.int32(y_lines),
        cp.float32(temperature), cp.float32(J), cp.int32(precision), cp.int32(frames)
    ))

    # --- 渲染步 ---
    render_spin_field(grid_render, block_dim, (
        spin_field, image_buffer,
        cp.int32(x_words_num), cp.int32(y_lines)
    ))

    # --- 显示逻辑 ---
    # 每运算几帧显示一次，可以提高观察速度 (设为 1 则每帧渲染)
    if frames % 1 == 0: 
        # .get() 会将数据从显存拷回 CPU 内存并转为 Numpy Array
        img_np = image_buffer.get()
        cv2.imshow("Ising 2D Simulation", img_np)
        
        # 捕获键盘
        key = cv2.waitKey(1)
        if key == 27: # 27 是 ESC 键的 ASCII 码
            break
            
    frames += 1

cv2.destroyAllWindows()