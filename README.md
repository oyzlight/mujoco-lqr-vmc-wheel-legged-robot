# MuJoCo LQR + VMC 轮腿机器人控制

使用 **MuJoCo + LQR + VMC** 控制一个四髋驱动、双轮驱动的闭环轮腿机器人，支持前后移动、左右旋转、倒地后电机起身、腾空收腿跳跃，以及固定场景中的 **15 cm 台阶上阶**。

每侧腿有两个独立髋部输入和两个闭环；膝部为被动销轴，**没有膝电机**。仿真只向四个髋电机、两个轮电机施加力矩。移动、跳跃和起身控制过程中不重写机器人位置或速度，也不施加机体外力。

## 视频演示

以下六段视频可直接在本页点击播放。[原始 MP4 与动图目录](media/)。录像均为实际控制器驱动的正常速度仿真，并检查结尾状态。

### 1. 前进、后退与停车

https://github.com/user-attachments/assets/62dc5c75-a430-4a0a-bd58-4cbfd09985e2

### 2. 左转、右转与原地旋转

https://github.com/user-attachments/assets/e88945bc-4f44-4d19-87c5-188407762554

### 3. 前倾倒地后电机起身

https://github.com/user-attachments/assets/b067bf93-f98f-421b-a0ad-271e13407a6e

### 4. 后倾倒地后电机起身

https://github.com/user-attachments/assets/7bdaae7c-49db-4d05-8a3b-7590090dcc90

### 5. 蹲伏、起跳、收腿与落地

https://github.com/user-attachments/assets/2f362bad-22f6-4857-86ba-418e5c473f95

### 6. 15 cm 台阶助跑跳跃

https://github.com/user-attachments/assets/5a87a7dc-c783-4f20-aaab-5c655ff1dddf

起身视频从预设的前倾 / 后倾 70° 倒地初始姿态开始，后续完全由电机力矩和接触动力学完成。验证还覆盖了前后 110° 的初始姿态。**完全侧躺和接近倒扣目前不支持**。

## 安装与运行

已验证环境：Windows、Python 3.11、MuJoCo 3.15.0。`requirements.txt` 固定了演示使用的依赖版本。建议创建独立虚拟环境。

```powershell
git clone https://github.com/oyzlight/mujoco-lqr-vmc-wheel-legged-robot.git
cd mujoco-lqr-vmc-wheel-legged-robot
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_keyboard.py
```

Windows 安装依赖后，也可以双击 `start_keyboard.cmd`。启动器优先使用项目中的 `.venv`；也可通过 `ROBOT_PYTHON` 指定解释器。首次启动会求静态平衡并设计多个腿长的 LQR，终端会显示初始化进度。

直接观看自动上阶：

```powershell
.venv\Scripts\python.exe run_keyboard.py --step-demo
```

或双击 `start_step_jump.cmd`。只在平地上操作：

```powershell
.venv\Scripts\python.exe run_keyboard.py --terrain flat
```

在其他系统使用对应的虚拟环境解释器路径；macOS 的 MuJoCo passive viewer 需要使用 `mjpython` 启动。窗口需要可用的图形环境。

## 键盘操作

点击仿真窗口使其获得焦点。速度采用逐次按键设定，**松键后保持设定值，按 X 停车**。

| 按键 | 动作 |
| --- | --- |
| ↑ / ↓ | 目标前后速度增加 / 减少 0.15 m/s；默认上限 ±0.4 m/s |
| ← / → | 左 / 右转角速度增加 0.5 rad/s；上限 ±1.2 rad/s |
| X | 减速停车，取消待执行的跳跃或上阶助跑 |
| 空格 | 普通跳跃；保留当前前后速度，支持前跳、后跳和原地跳 |
| J | 从出生点附近执行已验证的 15 cm 上阶辅助；建议先按 R 再按 J |
| G | 手动触发或重试电机起身；倒地时默认自动尝试 |
| R | 明确地将仿真重置到初始站姿 |
| Esc | 退出窗口 |

两轮结构不支持直接横向平移；向左右移动需要转向后前进。跳跃期间锁定前后速度目标，转向先停止。镜头跟随水平位置，观察高度固定，保留地面参照。

自动仿真重置默认关闭。`--auto-reset` 可显式启用起身失败后的重置，`--reset-delay 1.5` 控制等待时间；`--no-self-righting` 可关闭自动电机起身。**电机起身与 R 重置是两种不同的操作。**

## 控制方法

- **闭环运动学**：两次两圆求交得到连接点，按参考装配分支求轮轴位置；对闭环约束解析求导得到雅可比。
- **LQR**：在真实接触模型的静态平衡点附近数值线性化，控制前后运动和俯仰平衡；状态为轮轴位置、腿倾角、机身俯仰及各自速度。
- **VMC**：控制虚拟腿长和腿角，通过闭环雅可比把虚拟力 / 力矩映射到两个髋输入。外部 1:1 齿轮反向，因此电机输入力矩为 `-J.T @ [F, M]`。
- **横滚 / 航向**：左右腿差动支撑和左右轮差动力矩。
- **跳跃**：平衡 → 蹲伏 → 蹬伸 → 腾空收腿 → 落地准备 → 缓冲 → 恢复。
- **起身**：收腿撑地 → 伸腿 → LQR 接管制动 → 检查双轮承重、机身离地并稳定 0.4 秒。超时或不支持的姿态保留失败状态。

物理步长 **0.5 ms**，控制更新周期 **5 ms / 200 Hz**。所有实际电机力矩继续经过近似 24 V 力矩—转速包络：单个髋电机峰值 **7 N·m**，单个轮电机峰值 **2 N·m**。这些是模组输出端参数，不再额外乘内部减速比。

详细说明见 [控制与坐标约定](docs/CONTROL.md)、[15 cm 上阶研究记录](md/STEP15_JUMP_REPORT.md)。

## 15 cm 上阶的范围

场景只有一个台阶：前沿 **X=1.5 m**，顶面 **Z=0.15 m**，踏面长 2 m、宽 1.3 m。

J 模式从出生点沿 +X 使用 0.6 m/s 的独立助跑目标，在轮轴 X≈0.73 m 开始准备，采用 8 cm 深蹲和 8 cm 收腿目标。越过立面后先短腿落地，再撑到 18 cm 腿长并制动。原点及前后偏移 2 cm 的三组测试均完成上阶，轮底最高约 **18.7 cm**，站稳后轮轴高约 **19 cm**、机身根部高约 **38 cm**。

该辅助针对当前已知场景，不包含台阶识别、路径导航或任意台阶的自动规划。普通空格使用自由跳跃轨迹。

## 复核与录制

```powershell
# 普通移动、两次跳跃及落地恢复
python run_keyboard.py --smoke-test --terrain flat

# 转向、转向后的移动与跳跃
python run_keyboard.py --turn-test --terrain flat

# 实际 J 键路径，检查双轮在台阶顶面站稳
python tools/validate_step_jump.py

# 前后倒地后的电机起身，检查控制过程中没有改写 qpos/qvel
python tools/validate_self_righting.py

# 模型闭环、CAD 对齐、雅可比与六个电机接线
python tools/validate_closed_model.py

# 重录本 README 的所有视频与动图
python tools/record_demos.py
```

[视频复核记录](media/demo_report.json)与[上阶复核报告](results/step15_validation.json)、[起身复核报告](results/self_righting_validation.json)随仓库提供。重新运行检查会更新本地结果。

## 主要文件

```text
closed_kinematics.py       双输入双闭环正解、逆解和解析雅可比
lqr_vmc.py                静态平衡、线性化、LQR、VMC、差动控制
md/jump_controller.py     普通跳跃状态机
step_jump.py              15 cm 已知台阶的上阶轨迹与助跑辅助
self_righting.py          电机起身控制
motor_limits.py           近似电机力矩—转速限制
run_keyboard.py           键盘仿真入口
robot*.xml                自由机体、平面、固定机架与台阶场景
closed_meshes/            运行所需的 19 组刚性网格
mechanism_analysis/       修正后的几何点数据
tools/                    验证、参数对比与视频录制
media/                    MP4、GIF 与复核记录
```

## 模型边界

这是当前估计质量、惯量与电机包络下的仿真控制研究，**不是实机性能认证**。质量 / 惯量未实测辨识，力矩—转速曲线为工程近似；未覆盖杆件自碰撞、实物机械止挡、电池压降与热限制。起身仅验证了平地的部分前后倾倒姿态，上阶仅验证了固定的 15 cm 场景。仓库包含运行网格与几何数据，未包含原始 CAD 装配文件。
