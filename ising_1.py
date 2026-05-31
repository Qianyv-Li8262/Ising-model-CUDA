import cupy as cp
import cv2
import numpy as np
import os,glfw,time,sys
from zero_copy_window import *
base_path = os.path.dirname(os.path.abspath(__file__))
kernel_path = os.path.join(base_path, "ising_model_pipeline.cu")
with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()

module = cp.RawModule(code=cuda_source, options=('-use_fast_math','-std=c++17'))
isingstep = module.get_function('isingstep')
render_spin_field = module.get_function('render_spin_field')
initrng = module.get_function('initrng')
# ========== 3. 初始化参数 ==========
# 定义画面大小 (需保证宽度是 32 的倍数)
width_pixels = 2048
height_pixels = 2048
x_words_num = width_pixels // 32
y_lines = height_pixels
temperature = 1.9
# 物理参数
J = 1.0
precision = 12 # 概率掩码精度

# 初始化随机自旋场和图像缓冲区
# 随机生成无符号 32 位整数
spin_field = cp.random.randint(0, 0xFFFFFFFF, size=(y_lines, x_words_num), dtype=cp.uint32)
# 分配图像显存 (注意这里形状：高度, 宽度, 通道数4)
# image_buffer = cp.zeros((y_lines, width_pixels, 4), dtype=cp.uint8)
tot_energy = cp.zeros(1, dtype=cp.int32)

# 网格与线程块配置
block_dim = (32, 8) 
# isingstep 网格大小（处理 word）
grid_ising = ((x_words_num + block_dim[0] - 1) // block_dim[0], 
              (y_lines + block_dim[1] - 1) // block_dim[1])
# render 网格大小（处理 pixel）
grid_render = ((width_pixels + block_dim[0] - 1) // block_dim[0], 
               (y_lines + block_dim[1] - 1) // block_dim[1])
window = ZeroCopyWindow(width_pixels,height_pixels,'ising')
frames=1
steps = 1
import matplotlib.pyplot as plt

plt.ion()  # 开启交互模式（非阻塞）
fig, ax = plt.subplots(figsize=(6, 4))
fig.canvas.manager.set_window_title('Energy Monitor')
line, = ax.plot([], [], 'b-', linewidth=2)  # 初始化一条空折线
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Total Energy")
ax.set_title(f"Ising Model Energy (T={temperature:.3f})")
ax.grid(True)

# 用于存储图表数据的列表
history_time = []
history_energy = []
current_plot_time = 0.0  # X轴时间（每次重置归零）

state = cp.zeros((height_pixels,width_pixels),dtype=cp.uint64)
# inc = cp.zeros((height_pixels,width_pixels),dtype=cp.uint64)
initrng(grid_ising, block_dim,(state,cp.int32(x_words_num), cp.int32(y_lines)))

start_time = time.time()
tot_steps = 0

cp.cuda.profiler.start()

while not window.should_close():
    for _ in range(steps):
        tot_energy.fill(0)
        isingstep(grid_ising, block_dim, (
        spin_field, tot_energy, cp.bool_(True),
        cp.int32(x_words_num), cp.int32(y_lines),
        cp.float32(temperature), cp.float32(J), cp.int32(precision), cp.int32(frames),state
        ))
    
    # 第二步：更新黑格子
        isingstep(grid_ising, block_dim, (
        spin_field, tot_energy, cp.bool_(False),
        cp.int32(x_words_num), cp.int32(y_lines),
        cp.float32(temperature), cp.float32(J), cp.int32(precision), cp.int32(frames),state
        ))
        frames+=1
    

    
    tot_steps+=steps
    frame=window.map_pbo()
    render_spin_field(grid_render, block_dim, (
        spin_field, frame,
        cp.int32(x_words_num), cp.int32(y_lines)
        ))
    window.unmap_and_draw()
    cp.cuda.Device().synchronize()
    # if frames==3:
    #     cp.cuda.profiler.stop()
    #     sys.exit()
    temp_changed = False
    if glfw.KEY_W in window.key_pressed:
        temperature += 0.001
        temp_changed = True
    if glfw.KEY_S in window.key_pressed:
        temperature -= 0.001
        temp_changed = True
        
    if temp_changed:
        # 重置图表数据
        history_time.clear()
        history_energy.clear()
        current_plot_time = 0.0
        ax.set_title(f"Ising Model Energy (T={temperature:.3f})")

    now = time.time()
    elapsed = now - start_time
    
    if elapsed > 0.2:  # 每秒约 5 次更新显示
        actual_fps = tot_steps / elapsed
        current_e = tot_energy.get()[0] * J
        print(f"\rTemperature: {temperature:.3f} | Steps/Sec: {actual_fps:.0f} | FPS: {actual_fps/steps:.0f} | Energy: {current_e:.0f}    ", end="")
        
        # --- 新增：更新图表 ---
        current_plot_time += elapsed
        history_time.append(current_plot_time)
        history_energy.append(current_e)
        
        # 可选：限制历史数据长度（例如最多保留最近 300 个点，约 60 秒），避免内存无限涨
        if len(history_time) > 300:
            history_time.pop(0)
            history_energy.pop(0)
            
        # 更新折线数据
        line.set_data(history_time, history_energy)
        
        # 自动调整坐标轴缩放
        ax.relim()
        ax.autoscale_view()
        
        # 刷新画布表面（关键：由于是非阻塞，必须手动 flush_events）
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        
        # 重置统计
        start_time = now
        tot_steps = 0
