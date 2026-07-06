# Minesweeper RL Agent

基于 Gymnasium + PyTorch 的扫雷强化学习项目，包含 DQN、PPO、简化 GRPO、行为克隆和一个 C++ 纯文本扫雷小游戏。

当前模型使用**全卷积空间策略头**：网络在输出动作前保留 `rows x cols` 的格子位置特征，再为每个格子生成动作 logits/Q 值。相比把整块棋盘压成全局向量，这种结构更适合扫雷这种“每个格子都是一个动作”的任务。

## 项目结构

```text
mine-clear-llm/
├── mine_env.py          # Gymnasium 扫雷环境
├── models.py            # 空间 CNN 特征提取器 + DQN/PPO/GRPO 网络
├── agents.py            # DQN/PPO/GRPO Agent 实现
├── train.py             # 强化学习训练脚本
├── train_bc.py          # 行为克隆训练脚本
├── expert.py            # 规则专家/演示数据生成
├── play.py              # 交互播放/批量评估脚本
├── minesweeper.cpp      # C++ 扫雷游戏（纯文本 ASCII）
├── tests/               # 基础回归测试
├── requirements.txt     # Python 依赖
└── Makefile             # 常用训练、恢复、评估命令
```

## 安装

```bash
python3 -m pip install -r requirements.txt
```

运行回归测试：

```bash
make test
```

编译并运行 C++ 文本版扫雷：

```bash
make minesweeper
./minesweeper
```

## 最常用工作流

### 1. 先跑 4x4 小棋盘

建议先确认小棋盘能学会，再切到 9x9。

```bash
make train-grpo EPISODES=20000 \
    ROWS=4 \
    COLS=4 \
    MINES=2 \
    RUN_NAME=grpo_spatial_real_4x4_2m \
    SAVE_DIR=checkpoints/grpo_spatial_real_4x4_2m \
    DEVICE=cuda \
    EXTRA_ARGS="--group_size 32 --batch_size 128"
```

训练开始时应看到类似：

```text
Training GRPO on 4x4 with 2 mines
Action mode: reveal; actions: 16; total safe cells: 14
```

如果是 4x4 / 2 雷，`Safe` 最大只能是 `14`。如果日志里 `Safe` 到了 40 多，说明你实际仍在跑 9x9，而不是 4x4。

### 2. 评估 4x4 checkpoint

```bash
make eval-grpo \
    SAVE_DIR=checkpoints/grpo_spatial_real_4x4_2m \
    ROWS=4 \
    COLS=4 \
    MINES=2 \
    GAMES=100
```

看一局具体操作：

```bash
make play-grpo \
    SAVE_DIR=checkpoints/grpo_spatial_real_4x4_2m \
    ROWS=4 \
    COLS=4 \
    MINES=2 \
    GAMES=1
```

### 3. 训练 9x9 / 10 雷

不要复用 4x4 的目录，给 9x9 单独开 `RUN_NAME` 和 `SAVE_DIR`：

```bash
make train-grpo EPISODES=100000 \
    ROWS=9 \
    COLS=9 \
    MINES=10 \
    RUN_NAME=grpo_spatial_9x9_10m \
    SAVE_DIR=checkpoints/grpo_spatial_9x9_10m \
    DEVICE=cuda \
    EXTRA_ARGS="--group_size 64 --batch_size 256"
```

评估：

```bash
make eval-grpo \
    SAVE_DIR=checkpoints/grpo_spatial_9x9_10m \
    ROWS=9 \
    COLS=9 \
    MINES=10 \
    GAMES=500
```

## Makefile 参数说明

常用变量：

| 变量 | 作用 | 默认值 |
|------|------|--------|
| `EPISODES` | 训练总 episode 数 | `10000` |
| `ROWS` / `COLS` | 棋盘大小 | `9` / `9` |
| `MINES` | 地雷数，必须 `< ROWS * COLS` | `10` |
| `ACTION_MODE` | `reveal` 或 `reveal_flag` | `reveal` |
| `RUN_NAME` | TensorBoard 子目录名 | 空 |
| `SAVE_DIR` | checkpoint 保存/读取目录 | `checkpoints/<algo>` |
| `CHECKPOINT` | 播放/评估时显式指定 checkpoint 文件 | 空 |
| `DEVICE` | `cpu`、`cuda`、`cuda:0` 等 | 训练默认自动选 CUDA；play 默认 CPU |
| `GAMES` | 播放/评估局数 | `100` |
| `EXTRA_ARGS` | 传给 `train.py` 的额外参数 | 空 |

示例：

```bash
# DQN / PPO / GRPO 训练
make train-dqn
make train-ppo
make train-grpo

# 断点继续，加载 SAVE_DIR/latest.pt 和 training_state.json
make resume-grpo \
    EPISODES=300000 \
    ROWS=9 COLS=9 MINES=10 \
    SAVE_DIR=checkpoints/grpo_spatial_9x9_10m \
    RUN_NAME=grpo_spatial_9x9_10m \
    DEVICE=cuda \
    EXTRA_ARGS="--group_size 64 --batch_size 256"

# 注意：EPISODES 是“训练到第多少局”，不是“再训练多少局”。
```

播放/评估 checkpoint 的优先级：

1. 如果传了 `CHECKPOINT=...`，使用该文件；
2. 否则如果传了 `SAVE_DIR=...`，使用 `SAVE_DIR/best.pt`；
3. 否则使用默认 `checkpoints/<algo>/best.pt`。

例如：

```bash
# 用 SAVE_DIR/best.pt
make eval-grpo SAVE_DIR=checkpoints/grpo_spatial_real_4x4_2m ROWS=4 COLS=4 MINES=2

# 显式用 latest.pt
make eval-grpo \
    CHECKPOINT=checkpoints/grpo_spatial_real_4x4_2m/latest.pt \
    ROWS=4 COLS=4 MINES=2
```

## Checkpoint 和配置检查

训练目录会保存：

```text
checkpoints/<experiment>/
├── best.pt              # 最近 100 局胜率最优；胜率相同再比较 reward
├── latest.pt            # 最新 checkpoint
├── final.pt             # 训练结束 checkpoint
├── ep_<episode>.pt      # 周期性保存
├── run_config.json      # 棋盘、算法、动作模式等配置
└── training_state.json  # episode、best 指标、total steps 等
```

`play.py` 会检查 checkpoint 旁边的 `run_config.json`。如果你用 4x4 checkpoint 却按默认 9x9 参数播放，会直接报错并提示正确变量，例如：

```text
Use matching make variables, e.g. ROWS=4 COLS=4 MINES=2 ACTION_MODE=reveal
```

如确实想把 checkpoint 迁移到不同棋盘测试，可以直接调用 `play.py` 并传：

```bash
python3 play.py --algo grpo \
    --checkpoint checkpoints/grpo_spatial_real_4x4_2m/best.pt \
    --rows 9 --cols 9 --mines 10 \
    --eval \
    --allow_config_mismatch
```

通常不建议这么做，因为动作维度或棋盘分布不一致时效果没有保证。

## 环境定义

### 观测空间

```text
Box(low=0, high=1, shape=(5, rows, cols), dtype=float32)
```

5 个通道分别是：

1. 已揭开的格子；
2. 已标记/插旗的格子；
3. 已揭开数字，按 `number / 8` 归一化；
4. 已揭开数字周围的未揭开邻居数量，按 `/ 8` 归一化；
5. 边界未知格：未揭开且邻接至少一个已揭开格子。

### 动作空间

默认 `ACTION_MODE=reveal`：

```text
Discrete(rows * cols)
action = r * cols + c  # 揭开 (r, c)
```

可选 `ACTION_MODE=reveal_flag`：

```text
Discrete(2 * rows * cols)
0 <= action < rows*cols              # 揭开格子
rows*cols <= action < 2*rows*cols    # 切换插旗状态
```

环境支持：

- 首次 reveal 自动保证安全；
- Gymnasium seed 复现；
- iterative flood fill；
- 合法动作 mask：

```python
mask = env.valid_action_mask()
```

## 奖励设计

当前 `mine_env.py` 中的 shaped reward：

- 揭开安全格：`1.0 + new_revealed * 1.0 + progress * 2.0`
- 踩雷：`-10.0 + progress * 5.0`
- 获胜：`100.0`
- 无效操作：`-1.0`
- `reveal_flag` 模式下切换旗帜：`-0.05`

注意：`Avg Reward` 可能在没有胜利时也上升，因为模型只要揭开很多安全格就能得到较高 shaped reward。因此训练效果优先看：

```text
Win Rate > Progress > Safe > Avg Reward > Loss
```

`Loss` 在 PPO/GRPO 中不是监督学习式的“越低越好”指标，接近 0 或为负都不一定表示异常。

## 算法说明

### DQN

- 全卷积空间特征提取；
- Dueling DQN：每格 advantage + 全局 value；
- Double DQN target：online 网络选动作，target 网络估值；
- Experience Replay；
- Epsilon-greedy 探索；
- target 计算支持 next-state 合法动作 mask。

### PPO

- 全卷积 Actor-Critic；
- 每格 actor logits + 全局 critic value；
- GAE 从后向前递推；
- Clipped surrogate objective；
- 达到 `--update_steps` 后批量更新；
- 更新时复用采样时的合法动作 mask。

### GRPO / Group-normalized Policy Optimization

当前实现是适配扫雷的简化版组相对优化：

- 收集一组 episode；
- 使用每一步 discounted return-to-go 做 credit assignment，避免把“最后踩雷”也按整局高回报强化；
- 对 advantage 做标准化；
- 使用 old log-prob、ratio 和 clip objective 更新；
- value head 使用标准化后的 return target；
- 默认 `--grpo_vf_coef 0.05`，避免 value loss 主导策略更新。

如果需要严格 LLM-style GRPO，需要进一步保证同一 group 内样本来自相同初始局面/同一 prompt；当前实现更接近“组归一化策略梯度”。

## 训练指标和 TensorBoard

启动 TensorBoard：

```bash
tensorboard --logdir runs
```

训练中每 100 局打印/记录：

- `Reward/avg_100`：最近 100 局平均 shaped reward；
- `WinRate/avg_100`：最近 100 局胜率；
- `Progress/avg_100`：最近 100 局平均揭开的安全格比例；
- `SafeCells/avg_100`：最近 100 局平均揭开的安全格数；
- `Loss/latest`：最近一次优化 loss；
- DQN 额外记录 `Epsilon`。

示例解读：

```text
Episode 100000 | Avg Reward: 49.53 | Win Rate: 0.0% | Progress: 67.4% | Safe: 47.9 | Loss: 0.0098
```

如果是 9x9 / 10 雷，总安全格是 `71`，`Safe: 47.9` 约等于 `67.4%` 进度。这说明模型能打开很多安全格，但没有完成整局；不是程序崩了，而是策略仍然不会收盘。

## GPU 使用说明

训练可显式指定 GPU：

```bash
make train-grpo DEVICE=cuda
```

或直接：

```bash
python3 train.py --algo grpo --device cuda
```

注意：当前环境采样是 Python 单环境逐局循环，瓶颈经常在环境交互，而不是神经网络。因此即使 `DEVICE=cuda`，`nvidia-smi` 中 GPU 利用率也可能不高。真正提升 GPU 利用率需要后续加入多进程/向量化环境采样。

## 行为克隆

```bash
make train-bc
```

或：

```bash
python3 train_bc.py --n_demos 5000 --epochs 20
```

行为克隆可作为后续 RL warm start 的基础，但当前 Makefile 的主流程仍以 DQN/PPO/GRPO 为主。

## 常见问题

### `FileNotFoundError: checkpoints/grpo/best.pt`

说明你用的是默认播放路径，但训练时可能用了自定义 `SAVE_DIR`。

解决：

```bash
make eval-grpo SAVE_DIR=你的训练目录 ROWS=... COLS=... MINES=...
```

或显式指定文件：

```bash
make eval-grpo CHECKPOINT=你的训练目录/latest.pt ROWS=... COLS=... MINES=...
```

### RUN_NAME 写了 4x4，但 Safe 还是 48 左右

`RUN_NAME` 只是日志名字，不改变棋盘大小。必须传：

```bash
ROWS=4 COLS=4 MINES=2
```

4x4 / 2 雷总安全格只有 14，`Safe` 不可能是 48。

### `MINES=40 ROWS=4 COLS=4` 报错

4x4 只有 16 格，地雷数必须小于 16，最多 15。但实际训练中还要留足安全格，推荐：

```bash
ROWS=4 COLS=4 MINES=2
```

### `^Z` 后训练停住了

`^Z` 是暂停任务，不是退出。查看：

```bash
jobs
```

恢复：

```bash
fg %任务号
```

杀掉：

```bash
kill %任务号
```

### 旧 checkpoint 不能 resume

当前模型已经切到 `spatial_v1` 空间卷积结构，旧的 pooled-model checkpoint 不兼容。请新开 `SAVE_DIR` 训练，不要从旧目录 resume。

## 清理

```bash
make clean
```

会删除：

```text
minesweeper
checkpoints/
runs/
.pytest_cache/
```

注意：`make clean` 会删除所有训练 checkpoint 和 TensorBoard 日志，执行前请确认不再需要。
