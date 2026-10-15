# FPS-RL（实验分支）

轻量级 3D 第一人称射击 + 强化学习实验环境。纯 Python（Panda3D + NumPy）。

本分支只做实验环境：

- **地图**：沙漠灰 `desert2`（CS 1.6 de_dust2 几何与贴图）。
- **模式**：只有歼灭。清空对方即赢下回合；时间耗尽时存活人数多的一方获胜，人数相同算平局。每回合全员满状态重生，没有经济和购买。参数默认值：冻结时间 5 秒、回合 115 秒、结算 5 秒、先赢 13 回合，12 回合后换边。
- **武器方案**：固定 3 套，每套都带格洛克、军刀、高爆手雷、烟雾弹、闪光弹和燃烧弹。

| 方案 key | 主武器 |
|---|---|
| `rifle` | AK-47（步枪） |
| `sniper` | AWM（狙击枪） |
| `shotgun` | Nova（散弹枪） |

  每局开始时 bot 各自随机分到一套。玩家可以用 `--loadout` 指定，冻结时间内按 B 切换（冻结时间立即生效，否则下回合生效）。
- **没有主菜单 GUI**，只保留两种入口：人操作和观战。

## 安装

```bash
pip install -r requirements.txt
```

已在 Windows + Python 3.14 + Panda3D 1.10.16 上验证。

## 运行

```bash
py play.py                                         # 人操作：你 + bot 对阵 bot（默认 4v4）
py play.py --loadout sniper --difficulty hard --team-size 5
py play.py --spectate                              # 观战：全员 bot，默认上帝视角
py play.py --autoplay                              # 本地玩家由 bot 控制，第一人称跟随
py play.py --win-rounds 5 --seed 0 --size 1920x1080 --graphics high
py play.py --spectate --screenshot shot.png --shot-time 8   # 截图后退出
```

**人操作按键**

| 按键 | 功能 |
|---|---|
| WASD / 空格 / Ctrl 或 C / Shift | 移动 / 跳 / 蹲 / 静步 |
| Q / E | 左 / 右探头 |
| 左键 / 右键 | 开火、轻刀、捏雷投掷 / 瞄准、开镜、重刀、低抛 |
| R / 1-7 / X / 滚轮 | 换弹 / 选武器 / 上一把 / 切换 |
| B | 切换武器方案 |
| F | 检视 |
| Tab / Esc | 计分板 / 释放鼠标 |
| F1 / F2 / F3 / F4 / F5 | 帮助 / 瞄准按住↔切换 / 第三人称 / FPS / 重开 |
| +/- | 灵敏度 |

**观战按键**：WASD 和鼠标飞行，Q / E 升降，Shift 加速；F 在上帝视角和第一人称之间切换，第一人称下左键 / 右键切换观战对象；Tab 计分板，F5 重开。

## RL 接口

```python
import fpsrl, gymnasium as gym

env = fpsrl.FPSEnv(team_size=4, learner_loadout="sniper", bot_loadout="random")
env = gym.make("FPSRL-Elim-5v5-v0")         # 也可以用 FPSRL-Elim-v0
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

| Gym id | 说明 |
|---|---|
| `FPSRL-Elim-v0` | 4v4 歼灭，默认参数 |
| `FPSRL-Elim-5v5-v0` | 5v5，先赢 5 回合，4 回合后换边 |

学习者是 `learner_team` 队的 0 号位，其余位置由规则 bot 填充。

**动作**：默认 `action_type="heads"`，是一个由多个 `Discrete` 头组成的 `Dict`，包括 `move_direction`、`move_way`、`yaw`、`pitch`、`body`、`fight`、`target_enemy`、`thrown_move_direction`，以及保留不用的 `throw_target`、`predict_interval`、`path_goal`。`action_type="multidiscrete"` 是 11 维 `MultiDiscrete`，依次为前后、左右、yaw、pitch、开火、副键、跳、蹲、换弹、切枪、探头。`action_type="box"` 语义相同，但转向是连续的。

**观测**：默认 `obs_mode="vector"`，返回数值 `Dict`，包括自身 / 敌人 / 队友状态（110 维，自身状态里含当前武器方案的 one-hot）、武器状态、lidar、相机和投掷物。`"flat"` 是 162 维向量，`"rgb"` 返回离屏画面，`"both"` 两者都返回。

**奖励**：`RewardConfig` 包含 `kill`、`headshot_bonus`、`death`、`damage_dealt`、`damage_taken`、`team_damage`、`win`、`lose`、`round_win`、`round_lose` 和 `step`。`info["events"]` 是本步的原始事件，可以用来自定义奖励。

### 常用 `EnvConfig` 字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `mode_params` | `{}` | 歼灭参数覆盖：`freeze_time` / `round_time` / `post_time` / `win_rounds` / `half_rounds` |
| `team_size` | 4 | 每队人数 |
| `bot_difficulty` | `"normal"` | `easy/normal/hard/insane` 或 0~1 |
| `learner_loadout` | `"random"` | `rifle` / `sniper` / `shotgun` / `random` |
| `bot_loadout` | `"random"` | 同上，作用于所有 bot（`random` 表示每个 bot 各自随机） |
| `max_episode_steps` | 2250 | 截断步数（4 tick/步） |
| `end_on_death` / `friendly_fire` | False / False | |

### 多智能体

```python
env = fpsrl.FPSMultiAgentEnv(team_size=2, controlled=[0, 2])
```

### 底层 API

```python
from fpsrl import World, MatchConfig, Action
w = World(MatchConfig(team_size=2, loadouts=["rifle", None, "sniper", None], mode_params={"freeze_time": 0}))
w.reset(seed=0)
events = w.step({0: Action(move_forward=1)})
w.set_loadout(w.agents[1], "shotgun")   # 冻结时间内立即换装
```

`MatchConfig.loadouts` 里为 `None` 的位置，每局随机分配方案。

## 示例与测试

```bash
py examples/random_agent.py [--render] [--obs rgb]
py examples/train_ppo.py --envs 8 [--loadout sniper]
py examples/multiagent_selfplay.py --render
py tools/bot_match.py 3 4 normal          # 无头 bot 对局：分钟数 每队人数 难度，按方案统计击杀
py -m pytest tests
```

## 代码结构

```
fpsrl/
  config.py weapons.py   物理常量、武器参数、固定武器方案
  sim.py mode.py         权威模拟、歼灭模式
  geometry.py nav.py     射线/遮挡、导航图
  bots.py                规则 bot
  maps/                  沙漠灰
  obs.py envs.py         观测、Gymnasium / 多智能体环境
  policy.py              PPO 策略网络
  render/                Panda3D 渲染、HUD、音效、本地对局（game.py）
play.py                  入口（人操作 / 观战）
tools/import_dust2.py    重新导入 de_dust2 网格
tools/top_view.py        地图俯视图
```

沙漠灰模型：[CS 1.6 de_dust2](https://sketchfab.com/3d-models/de-dust2-cs-map-056008d59eb849a29c0ab6884c0c3d87) by vrchris，CC-BY 4.0。
