# FPS-RL：运输船

轻量级 3D 第一人称射击游戏 + 强化学习环境。纯 Python（Panda3D + NumPy），模型、贴图、音效全部程序化生成，无外部美术资源。地图参考 CF「运输船」：两端出生平台、中间集装箱与木箱构成的多条通道、两侧舷墙、上层集装箱高点。

- **武器**：AK-47（全自动，固定后坐力图案）、AWM（开镜狙击，开火后退镜）、沙漠之鹰（半自动）、M9 军刀（轻击/重击/背刺）、高爆手雷、烟雾弹（可遮挡视线和 bot 感知）
- **手感**：60 Hz 固定步长模拟 + 渲染插值，鼠标视角每帧直接作用（零延迟）；Quake 式加速/摩擦移动、蹲跳、台阶；后坐力 punch 回复、移动/空中扩散；命中标记、伤害方向指示、动态准星
- **画面**：实时阴影、法线贴图、MSAA、渐变天空/云、动态海面、港口远景、粒子（枪口火焰、火花、血雾、爆炸、烟雾）、弹道、弹孔
- **RL**：Gymnasium 单智能体环境 + PettingZoo 风格多智能体环境；向量观测 / RGB 观测 / 两者兼有；无头模式约 1000 tick/s（8 人，含 bot）；可复现

## 安装

```bash
pip install -r requirements.txt
```

Python ≥ 3.10，在 Windows + Python 3.14 + Panda3D 1.10.16 上验证。首次运行会在 `fpsrl/assets/sfx` 生成音效缓存。

## 游玩

```bash
py play.py                              # 主菜单（背景是实时 bot 对战）
py play.py --quick                      # 跳过菜单直接开一局 4v4
py play.py --quick --team-size 2 --difficulty hard --primary awm
py play.py --autoplay                   # 观战（本地玩家也由 bot 控制）
py play.py --size 2560x1440 --fullscreen --msaa 8
```

主菜单：

- **单人训练**：人数 / 难度 / 阵营 / 主武器 / 击杀目标 / 时长
- **创建房间**：局域网主机（权威服务器）；空位由 bot 补齐，玩家加入时顶替 bot、离开后 bot 接管；对局结束 10 秒后自动开下一局
- **加入房间**：UDP 广播自动发现局域网房间，也可手动输入 IP:端口；可选偏好阵营
- **武器仓库**：为 AK-47 / AWM / 沙鹰 / 军刀 / 手雷选择皮肤（普通 / 稀有 / 史诗 / 传说），3D 旋转预览；其他玩家能看到你的皮肤
- **设置**：名称、鼠标灵敏度（滑条或 −/+，范围 0.1–12；游戏中也可按 +/−）、音量、开镜方式、FPS 显示、全屏；保存在 `~/.fpsrl/profile.json`

游戏中 Esc 打开暂停菜单（调灵敏度 / 音量、返回主菜单）。

### 局域网联机

主机「创建房间」后，同一网段的玩家在「加入房间」里即可看到。默认端口 TCP 47777（游戏）+ UDP 47778（房间发现）。首次运行时 Windows 防火墙会弹窗，请允许 Python 访问**专用网络**；跨网段时广播不可达，用手动 IP 连接。

网络模型：60 Hz 权威服务器 + 客户端预测/回滚重放（自己的移动零延迟）+ 其他玩家 50 ms 插值 + 服务器延迟补偿（命中判定回溯到客户端看到的时刻，最多 250 ms）。协议为长度前缀 JSON（不使用 pickle，对端无法执行代码）。

### 击杀反馈与检视

- 击杀时屏幕底部弹出击杀勋章（普通 / 爆头 / 刀杀 / 手雷），连杀显示 双杀 / 三杀 / 四杀 / 五杀 / 超神 和连杀点数，配合随连杀升调的击杀音效与爆头 / 刀杀 / 手雷专属音。
- 想换音效：把同名文件（如 `kill_1.wav` … `kill_6.wav`、`kill_hs.wav`、`kill_knife.wav`、`kill_nade.wav`，支持 wav/ogg/mp3）放进 `fpsrl/custom_sfx/` 即可覆盖。
- **F 键检视**：播放检视动作（枪械翻转展示侧面、军刀抛转），右下角显示「武器 | 皮肤名」和稀有度颜色；开火、换弹、开镜、切枪会打断检视。

| 按键 | 功能 |
|---|---|
| WASD / 空格 / Ctrl 或 C / Shift | 移动 / 跳 / 蹲 / 静步 |
| 左键 | 开火 / 轻刀 / 投掷 |
| 右键 | 瞄准 / 开镜 / 重刀 / 低抛 |
| R / 1-5 / Q / 滚轮 | 换弹 / 选武器 / 上一把 / 切换 |
| B | 主武器 AK ⇄ AWM（出生 6 秒内立即生效，否则下次出生） |
| F | 检视武器 / 皮肤 |
| Tab / Esc | 计分板 / 暂停菜单 |
| F1 / F2 / F3 / F4 / F5 | 帮助 / 瞄准按住↔切换 / 第三人称 / FPS / 重开 |
| +/- | 灵敏度 |

其它参数见 `py play.py -h`。

## RL 接口

### 单智能体

```python
import fpsrl

env = fpsrl.FPSEnv(team_size=4, bot_difficulty="normal")   # 或 gymnasium.make("FPSRL-TransportShip-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

学习者是 `learner_team` 队的 0 号位，其余所有位置由规则 bot 填充（`teammate_bots=False` 时队友不行动）。

**动作**（`action_type="multidiscrete"`，默认）：`MultiDiscrete([3,3,9,5,2,2,2,2,2,6])`

| 维度 | 含义 |
|---|---|
| 0, 1 | 前后 / 左右（0=负，1=停，2=正） |
| 2 | 水平转向（度/步）`[-30,-15,-6,-2,0,2,6,15,30]`，正 = 左转 |
| 3 | 俯仰（度/步）`[-10,-3,0,3,10]`，正 = 抬头 |
| 4-8 | 开火、副键（瞄准/重刀/低抛）、跳、蹲、换弹 |
| 9 | 切枪：0 不变，1-5 对应槽位（主武器、手枪、刀、手雷、烟雾） |

`action_type="box"`：`Box(-1,1,(10,))`，同样的语义，转向连续（`max_turn_per_step` 度）。每个 env step 执行 `action_repeat`（默认 4）个 60 Hz tick，转向均摊到每个 tick。

**观测**（`obs_mode="vector"`，默认 129 维，`[-5,5]` 归一化），索引区间可用 `env.obs_builder.layout()` 获取：

| 区块 | 维度 | 内容 |
|---|---|---|
| self | 34 | 存活、HP、位置、局部速度、朝向、武器槽 one-hot、弹药、换弹/开镜/就绪、雷数、后坐力、扩散、受击方向、剩余时间、比分差 |
| lidar | 32 | 水平 360° 射线到障碍物的距离 |
| enemies | 4×9 | 可见敌人（按距离）：相对位置、距离、到瞄准点的 yaw/pitch 误差 |
| allies | 3×5 | 队友相对位置、HP |
| projectiles | 2×6 | 最近的投掷物 |

`enemy_mode="all"` 提供全部敌人信息（特权观测，适合调试或 teacher 策略）。`obs_mode="rgb"` 返回第一人称离屏渲染 `(H,W,3) uint8`（`rgb_size=(128,96)`，约 110 步/秒），`"both"` 返回 `{"vector", "rgb"}`。

**奖励**由 `RewardConfig` 配置：击杀 +1、爆头额外 +0.25、死亡 −0.5、每 100 伤害 +0.5、每 100 承伤 −0.25、胜 +3 / 负 −3。`info["events"]` 包含本步所有原始事件（`shot`、`damage`、`kill`、`explode` 等），可以自定义奖励：

```python
env = fpsrl.FPSEnv(reward={"kill": 2.0, "death": -1.0, "step": -0.001}, end_on_death=True)
```

### 多智能体 / 自对弈

```python
env = fpsrl.FPSMultiAgentEnv(team_size=2, controlled=[0, 2])   # 两队各控一人，其余为 bot
obs, infos = env.reset(seed=0)                                 # {"agent_0": ..., "agent_2": ...}
obs, rew, term, trunc, info = env.step({a: env.action_space(a).sample() for a in env.agents})
```

`controlled=None` 时所有 `2*team_size` 名玩家都由外部控制。

### 常用 `EnvConfig` 字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `team_size` | 4 | 每队人数 |
| `bot_difficulty` | `"normal"` | `easy/normal/hard/insane` 或 0~1 浮点 |
| `bot_primaries` | `"mixed"` | bot 主武器：`ak47` / `awm` / `mixed` |
| `learner_primary` | `"ak47"` | 学习者主武器 |
| `kill_limit` / `time_limit` | 40 / 0 | 对局结束条件（0 = 不限） |
| `max_episode_steps` | 2250 | 截断步数（4 tick/步 ≈ 150 秒） |
| `end_on_death` | False | 学习者死亡即结束 episode |
| `respawn` / `friendly_fire` | True / False | |
| `render_realtime` | True | `render_mode="human"` 时按真实时间播放 |

### 底层 API

`fpsrl.World` 是与渲染无关的权威模拟，可以直接驱动做更灵活的实验（课程学习、自定义场景、脚本化测试）：

```python
from fpsrl import World, MatchConfig, Action
from fpsrl.bots import Bot

w = World(MatchConfig(team_size=1, primaries=["awm", "ak47"]))
w.reset(seed=0)
bot = Bot(w, 1, "hard")
events = w.step({0: Action(move_forward=1, look_yaw=0.5, fire=True), 1: bot.act()})
```

`World.step` 返回事件列表，`World.agents` 暴露所有玩家状态，`World.lidar` / `World.visibility` / `World.trace_bullet` 可用于构造新的观测。

## 示例与测试

```bash
py examples/random_agent.py [--render] [--obs rgb]   # 随机策略 + 吞吐测试
py examples/train_ppo.py --envs 8                    # stable-baselines3 PPO 基线
py examples/train_ppo.py --eval runs/ppo/final.zip   # 观看训练好的策略
py examples/multiagent_selfplay.py --render          # 多智能体骨架
py tests/test_sim.py                                 # 或 pytest tests
py tests/test_net.py                                 # 联机协议（本机回环，无渲染）
py tools/snapshots.py shots                          # 渲染一组截图检查画面
py tools/menu_shots.py                               # 菜单各页面截图
py tools/lan_test.py host  /  py tools/lan_test.py client   # 两个进程本机联机冒烟测试（bot 驱动）
```

`ClientSession(..., autoplay=True)` 让客户端由 bot 驱动，可作为「远程智能体」接入主机的模板。

## 代码结构

```
fpsrl/
  config.py  weapons.py        物理常量、武器参数（伤害、射速、后坐力图案……）
  sim.py                       权威模拟：移动/碰撞、射击、近战、投掷物、伤害、重生、事件
  geometry.py                  向量化射线-AABB、球体遮挡
  maps/transport_ship.py       地图（AABB 盒子 + 道具 + 出生点 + bot 战略点）
  bots.py                      导航网格 A*、感知、反应、瞄准噪声、压枪、投雷
  obs.py  envs.py              观测构造、Gymnasium / 多智能体环境
  render/                      Panda3D 渲染：场景、模型、贴图、特效、音效、第一人称手臂、HUD
    app.py menu.py             主菜单 / 暂停菜单 / 应用状态机
    game.py                    本地对局、局域网主机、客户端会话（预测 + 插值）
    skins.py killbanner.py     程序化皮肤、击杀勋章
  net/                         局域网协议、主机（TCP + UDP 发现）、客户端
  profile.py                   玩家设置 / 皮肤存档
play.py                        游戏入口
```

新增地图：在 `fpsrl/maps/` 下返回一个 `MapData`，并在 `maps/__init__.py` 中注册。新增武器：在 `weapons.py` 添加 `WeaponSpec`，在 `render/models.py` 添加模型构建函数。
