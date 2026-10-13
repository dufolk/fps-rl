# FPS-RL

轻量级 3D 第一人称射击游戏 + 强化学习环境。纯 Python（Panda3D + NumPy），模型、贴图、音效全部程序化生成，无外部美术资源。

- **地图 / 模式**

| 地图 | 模式 | 说明 |
|---|---|---|
| 运输船 `transport_ship` | 团队竞技 / 占点 | 参考 CF「运输船」：两端出生平台、集装箱通道、上层高点 |
| 沙漠灰 `desert2` | 爆破 / 团队竞技 | CS 1.6 de_dust2 原版几何与贴图（`tools/import_dust2.py`）：A 大 / A 小、中门、狗洞、B 洞、CT 出生 |
| 占点小镇 `hp_town` | 占点 / 团队竞技 | 由厘米坐标数据复刻（`tools/import_town.py`），A/B/C 三个点 |
| 银行 `bank` | 彩六 / 团队竞技 | 彩六银行：地下室 + 1F + 2F + 屋顶，可破坏墙 / 地板、加固、路障，4 组点位 |

- **模式规则**
  - **占点**：站在点内占领（人多更快），占有的点每秒加分，先到 200 分获胜。
  - **爆破**：CS 式经济（起始 $800、击杀奖励、连败补偿、安包 / 拆包奖励），冻结时间内在出生区按 B 购买；T 安放 C4，CT 拆除（拆弹器 5 秒，否则 10 秒），半场换边。
  - **彩六**：一条命；45 秒准备阶段（进攻方用无人机侦察，防守方加固墙体、给门窗上路障）；进攻方在一组点位中的一个放置拆解器；8 名干员各带专属装备（热切割、破门弹、大锤、闪光盾、装甲板、干扰器、阔剑、护甲包）。
- **武器**：AK-47、M4A4、M4A1、FAMAS、MP5、MP9、UZI、AWM、SCAR-20（连狙）、M249（大菠萝）、沙漠之鹰、双持贝雷塔、M9 军刀、高爆手雷、烟雾弹、闪光弹，以及彩六的破门炸药、铁丝网、可部署盾牌等。
- **手感**：60 Hz 固定步长模拟 + 渲染插值，鼠标视角每帧直接作用（零延迟）；Quake 式加速 / 摩擦移动、蹲跳、台阶；左右探头（Q/E，贴墙时自动收回）；第三人称下开镜切回第一人称瞄具；后坐力回复、移动 / 空中扩散；命中标记、伤害方向指示、动态准星。
- **画面**：实时阴影、法线贴图、MSAA、渐变天空 / 云、每张地图独立主题、粒子（枪口火焰、火花、血雾、爆炸、烟雾、碎墙）、弹道、弹孔。
- **RL**：Gymnasium 单智能体环境 + PettingZoo 风格多智能体环境，所有地图 / 模式都可训练；向量观测 / RGB 观测 / 两者兼有；可复现。

## 安装

```bash
pip install -r requirements.txt
```

Python ≥ 3.10，在 Windows + Python 3.14 + Panda3D 1.10.16 上验证。首次运行会在 `fpsrl/assets/sfx` 生成音效缓存。

## 游玩

```bash
py play.py                                      # 主菜单（背景是实时 bot 对战）
py play.py --quick                              # 跳过菜单直接开一局 4v4（运输船团队竞技）
py play.py --quick --map desert2                # 沙漠灰爆破（默认取地图的第一个模式）
py play.py --quick --map hp_town --mode dom --team-size 5
py play.py --quick --map bank --mode r6 --difficulty hard
py play.py --quick --primary scar20             # 无经济的模式里指定主武器
py play.py --autoplay                           # 观战（本地玩家也由 bot 控制）
py play.py --size 2560x1440 --fullscreen --msaa 8
```

主菜单：

- **单人训练**：地图 / 模式 / 人数 / 难度 / 阵营 / 主武器 / 击杀目标 / 时长
- **创建房间**：局域网主机（权威服务器），同样可选地图和模式；空位由 bot 补齐，玩家加入时顶替 bot、离开后 bot 接管；对局结束 10 秒后自动开下一局
- **加入房间**：UDP 广播自动发现局域网房间（列表显示地图和模式），也可手动输入 IP:端口；可选偏好阵营
- **武器仓库**：为各武器选择皮肤（普通 / 稀有 / 史诗 / 传说），3D 旋转预览；其他玩家能看到你的皮肤
- **设置**：名称、鼠标灵敏度（滑条或 −/+，范围 0.1–12；游戏中也可按 +/−）、音量、开镜方式、FPS 显示、全屏；保存在 `~/.fpsrl/profile.json`

游戏中 Esc 打开暂停菜单（调灵敏度 / 音量、返回主菜单）。

### 购买与干员

- **爆破模式**：冻结时间和开局 20 秒内、站在己方出生区时按 **B** 打开购买菜单（手枪 / 冲锋枪 / 步枪 / 狙击 / 重武器 / 投掷物，CT 额外可买拆弹器）。已拥有的显示为绿色，钱不够的变灰。
- **彩六模式**：准备阶段按 **B** 打开干员菜单，选择干员和主武器后立即换装；其他时间选择在下一回合生效。
- **其他模式**：B 循环切换主武器（出生 6 秒内立即生效，否则下次出生）。

### 彩六细节

- **回合**：45 秒准备 + 3 分钟行动，一条命；每回合从地图的 4 组点位中选一组（2 个点）；拆解器安放 7 秒、拆除 7 秒，安放后 45 秒内防守方拆除才算赢；先赢 4 回合，3:3 进入加时（先到 5），每 3 回合攻守互换。
- **破坏**：可破坏墙 / 地板被切成 0.75 m 的面板，子弹会穿透并打碎软墙，近战、手雷、破门炸药都能开洞；加固后的墙只有热切的热切割能炸开。墙洞、加固、路障都会实时更新碰撞、子弹遮挡和 bot 导航，并以位图同步给客户端。
- **侦察**：进攻方每人每回合 2 架无人机（准备阶段时人被冻结在出生点，只能开无人机），防守方共享地图上的摄像头；G 进入视角后画面带扫描线 / 噪点和 REC 标识，左键标记敌人（全队看到红色「!」标记 4 秒），摄像头视角右键切下一个；干扰器范围内无人机失灵。屏幕边缘还会显示当前点位 / 拆解器的方向和距离。
- **通用装备**：进攻方闪光弹 / 破门炸药 / 手雷 / 烟雾，防守方铁丝网（减速）+ 可部署盾牌；防守方站在门窗旁按住 F 上路障（可被子弹 / 近战 / 爆炸打碎）。

| 进攻 | 专属装备 | 防守 | 专属装备 |
|---|---|---|---|
| 热切 Thermite | 热切割，可炸开加固墙 | 城堡 Castle | 门窗防弹板 |
| 灰烬 Ash | 破墙弹，远程炸开软墙 | 禁言 Mute | 信号干扰器，屏蔽无人机和破门炸药 |
| 大锤 Sledge | 破墙锤，砸开软墙、地板、路障 | 卡普坎 Kapkan | 门窗绊雷 |
| 闪盾 Blitz | 闪光盾，正面防弹并致盲 | 战车 Rook | 护甲包，拾取后减伤 |

bot 在彩六里会：防守方准备阶段加固点位附近的墙、给门窗上路障；进攻方准备阶段开无人机沿导航图摸向点位并标记防守方，行动阶段用热切割 / 大锤 / 破门炸药开点位外墙，然后进点安放。导航图是多层的（地下室到屋顶，楼梯、跳下边缘），墙或地板打穿后路径会自动变短。

### 局域网联机

主机「创建房间」后，同一网段的玩家在「加入房间」里即可看到。默认端口 TCP 47777（游戏）+ UDP 47778（房间发现）。首次运行时 Windows 防火墙会弹窗，请允许 Python 访问**专用网络**；跨网段时广播不可达，用手动 IP 连接。

网络模型：60 Hz 权威服务器 + 客户端预测 / 回滚重放（自己的移动零延迟）+ 其他玩家 50 ms 插值 + 服务器延迟补偿（命中判定回溯到客户端看到的时刻，最多 250 ms）。墙体破坏、加固、模式状态（比分、点位、炸弹、回合）都会同步；购买和选干员作为命令发给主机执行。协议为长度前缀 JSON（不使用 pickle，对端无法执行代码）。

### 击杀反馈与检视

- 击杀时屏幕底部弹出击杀勋章（普通 / 爆头 / 刀杀 / 手雷），连杀显示 双杀 / 三杀 / 四杀 / 五杀 / 超神 和连杀点数，配合随连杀升调的击杀音效与爆头 / 刀杀 / 手雷专属音。
- 想换音效：把同名文件（如 `kill_1.wav` … `kill_6.wav`、`kill_hs.wav`、`kill_knife.wav`、`kill_nade.wav`，支持 wav/ogg/mp3）放进 `fpsrl/custom_sfx/` 即可覆盖。
- **F 键检视**：没有可交互对象时，F 播放检视动作，右下角显示「武器 | 皮肤名」和稀有度颜色；开火、换弹、开镜、切枪会打断检视。

| 按键 | 功能 |
|---|---|
| WASD / 空格 / Ctrl 或 C / Shift | 移动 / 跳 / 蹲 / 静步 |
| Q / E | 左 / 右探头（按住） |
| 左键 | 开火 / 轻刀 / 投掷 / 安包 |
| 右键 | 瞄准 / 开镜 / 重刀 / 低抛 |
| R / 1-7 / X / 滚轮 | 换弹 / 选武器（主、副、刀、雷、烟、装备、C4/拆解器）/ 上一把 / 切换 |
| F | 交互（按住）：拆包 / 安包、加固墙体、上路障；否则检视武器 |
| G | 彩六：进攻方无人机 / 防守方摄像头视角 |
| B | 购买菜单 / 干员菜单 / 切换主武器（见上） |
| Tab / Esc | 计分板 / 暂停菜单 |
| F1 / F2 / F3 / F4 / F5 | 帮助 / 瞄准按住↔切换 / 第三人称 / FPS / 重开 |
| +/- | 灵敏度 |

其它参数见 `py play.py -h`。

## RL 接口

### 单智能体

```python
import fpsrl
import gymnasium as gym

env = fpsrl.FPSEnv(team_size=4, bot_difficulty="normal")           # 运输船团队竞技
env = fpsrl.FPSEnv(map_name="desert2", mode="bomb", team_size=5)    # 任意地图 / 模式组合
env = gym.make("FPSRL-Bank-R6-v0")                                  # 或预注册的 id
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

| Gym id | 地图 / 模式 |
|---|---|
| `FPSRL-TransportShip-v0` | 运输船团队竞技（4v4） |
| `FPSRL-TransportShip-Dom-v0` | 运输船占点 |
| `FPSRL-Desert2-Bomb-v0` | 沙漠灰爆破（5v5，先赢 5 回合，4 回合后换边） |
| `FPSRL-HPTown-Dom-v0` | 占点小镇（5v5） |
| `FPSRL-Bank-R6-v0` | 银行彩六（5v5） |

学习者是 `learner_team` 队的 0 号位，其余所有位置由规则 bot 填充（`teammate_bots=False` 时队友不行动）。bot 会执行模式目标：占点、带包 / 安包 / 拆包、彩六加固和守点，爆破模式按经济自动购买。`info["score"]` 是当前比分（击杀数 / 占点分 / 回合胜场）。

**动作**（`action_type="multidiscrete"`，默认）：`MultiDiscrete([3,3,9,5,2,2,2,2,2,8,3,2,2])`

| 维度 | 含义 |
|---|---|
| 0, 1 | 前后 / 左右（0=负，1=停，2=正） |
| 2 | 水平转向（度/步）`[-30,-15,-6,-2,0,2,6,15,30]`，正 = 左转 |
| 3 | 俯仰（度/步）`[-10,-3,0,3,10]`，正 = 抬头 |
| 4-8 | 开火、副键（瞄准/重刀/低抛）、跳、蹲、换弹 |
| 9 | 切枪：0 不变，1-7 对应槽位（主武器、手枪、刀、手雷、烟雾、装备、C4/拆解器） |
| 10 | 探头：0 左，1 不探，2 右 |
| 11 | 交互（按住：安包 / 拆包 / 加固 / 路障） |
| 12 | 切换无人机 / 摄像头视角（彩六） |

`action_type="box"`：`Box(-1,1,(13,))`，同样的语义，转向连续（`max_turn_per_step` 度）。每个 env step 执行 `action_repeat`（默认 4）个 60 Hz tick，转向均摊到每个 tick。爆破模式下 `learner_auto_buy=True`（默认）让学习者在冻结时间按 bot 的策略自动购买。

**观测**（`obs_mode="vector"`，默认 156 维，`[-5,5]` 归一化），索引区间可用 `env.obs_builder.layout()` 获取：

| 区块 | 维度 | 内容 |
|---|---|---|
| self | 49 | 存活、HP、位置（按地图范围归一化）、局部速度、朝向、当前槽位 / 持有槽位、弹药、换弹 / 开镜 / 就绪、雷数、后坐力、扩散、受击方向、剩余时间、比分差、探头、交互进度、冻结、护甲、金钱、视角模式 |
| lidar | 32 | 水平 360° 射线到障碍物的距离 |
| enemies | 4×9 | 可见敌人（按距离）：相对位置、距离、到瞄准点的 yaw/pitch 误差 |
| allies | 3×5 | 队友相对位置、HP |
| projectiles | 2×6 | 最近的投掷物 |
| objective | 12 | 模式相关：占点为每个点的归属 / 进度 / 距离 / 方位；爆破 / 彩六为攻守方、回合阶段、剩余时间、炸弹状态 / 距离 / 方位、最近点位距离 / 方位、是否携带 |

`enemy_mode="all"` 提供全部敌人信息（特权观测，适合调试或 teacher 策略）。`obs_mode="rgb"` 返回第一人称离屏渲染 `(H,W,3) uint8`（`rgb_size=(128,96)`），`"both"` 返回 `{"vector", "rgb"}`。

**奖励**由 `RewardConfig` 配置：击杀 +1、爆头额外 +0.25、死亡 −0.5、每 100 伤害 +0.5、每 100 承伤 −0.25、胜 +3 / 负 −3；模式奖励：回合胜 +1 / 负 −1、己方占点 +0.5（不在点内 ×0.3）、丢点 −0.25、安包 +0.8、拆包 +0.8，另有 `hold`（每秒每个己方点）和 `in_objective`（每秒在目标区域内）两个默认为 0 的塑形项。`info["events"]` 包含本步所有原始事件（`shot`、`damage`、`kill`、`point_captured`、`bomb_planted`、`round_end`、`wall_break` 等），可以自定义奖励：

```python
env = fpsrl.FPSEnv(reward={"kill": 2.0, "death": -1.0, "step": -0.001}, end_on_death=True)
```

### 多智能体 / 自对弈

```python
env = fpsrl.FPSMultiAgentEnv(team_size=2, controlled=[0, 2])   # 两队各控一人，其余为 bot
obs, infos = env.reset(seed=0)                                 # {"agent_0": ..., "agent_2": ...}
obs, rew, term, trunc, info = env.step({a: env.action_space(a).sample() for a in env.agents})
```

`controlled=None` 时所有 `2*team_size` 名玩家都由外部控制。同样接受 `map_name` / `mode` / `mode_params`。

### 常用 `EnvConfig` 字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `map_name` / `mode` | `"transport_ship"` / `"tdm"` | 地图和模式（模式必须是该地图支持的） |
| `mode_params` | `{}` | 覆盖模式参数，如 `{"score_limit": 100}`、`{"win_rounds": 5, "freeze_time": 2}`、`{"prep_time": 20}` |
| `team_size` | 4 | 每队人数 |
| `bot_difficulty` | `"normal"` | `easy/normal/hard/insane` 或 0~1 浮点 |
| `bot_primaries` | `"mixed"` | bot 主武器：`ak47` / `awm` / `mixed`（步枪、冲锋枪、狙击混合） |
| `learner_primary` | `"ak47"` | 学习者主武器（无经济的模式） |
| `learner_auto_buy` | True | 爆破模式学习者自动购买 |
| `kill_limit` / `time_limit` | 40 / 0 | 团队竞技结束条件（0 = 不限） |
| `max_episode_steps` | 2250 | 截断步数（4 tick/步 ≈ 150 秒） |
| `end_on_death` | False | 学习者死亡即结束 episode |
| `respawn` / `friendly_fire` | True / False | 回合制模式强制不复活 |
| `render_realtime` | True | `render_mode="human"` 时按真实时间播放 |

### 底层 API

`fpsrl.World` 是与渲染无关的权威模拟，可以直接驱动做更灵活的实验（课程学习、自定义场景、脚本化测试）：

```python
from fpsrl import World, MatchConfig, Action
from fpsrl.bots import Bot

w = World(MatchConfig(team_size=2, map_name="desert2", mode="bomb", mode_params={"freeze_time": 1}))
w.reset(seed=0)
bots = [Bot(w, a.id, "hard") for a in w.agents[1:]]
events = w.step({0: Action(move_forward=1, lean=1), **{b.aid: b.act() for b in bots}})
w.buy(w.agents[0], "m4a1")        # 在购买区和购买时间内
print(w.score, w.mode.state())    # 比分、回合阶段、炸弹状态……
```

`World.step` 返回事件列表，`World.agents` 暴露所有玩家状态，`World.mode` 是当前模式（`obs(agent)` 给出 12 维目标观测），`World.destruct` 管理可破坏墙体，`World.lidar` / `World.visibility` / `World.trace_bullet` 可用于构造新的观测。

## 示例与测试

```bash
py examples/random_agent.py [--render] [--obs rgb] [--map desert2 --mode bomb]   # 随机策略 + 吞吐测试
py examples/train_ppo.py --envs 8 [--map hp_town --mode dom]   # stable-baselines3 PPO 基线
py examples/train_ppo.py --eval runs/ppo/final.zip             # 观看训练好的策略
py examples/multiagent_selfplay.py --render                    # 多智能体骨架
py tests/test_sim.py                                           # 或 pytest tests
py tests/test_modes.py                                         # 新武器、探头、占点 / 爆破 / 彩六、地图连通性、gym id
py tests/test_siege.py                                         # 破坏 / 加固 / 导航跟随破坏 / 彩六回合 / 无人机 / 摄像头 / 路障 / bot 对局
py tests/test_net.py                                           # 联机协议（本机回环，无渲染）
py tools/map_check.py bank r6                                  # 地图导航连通性检查
py tools/bot_match.py desert2 bomb 3                           # 无头 bot 对局，打印事件统计
py tools/interior_shots.py bank r6 shots/bank                  # 地图内部截图
py tools/ui_shots.py shots [r6]                                # 购买菜单 / 干员菜单截图
py tools/siege_shots.py shots att|def                          # 彩六 HUD：点位标记、无人机 / 摄像头画面
py tools/snapshots.py shots                                    # 渲染一组截图检查画面
py tools/menu_shots.py                                         # 菜单各页面截图
py tools/lan_test.py host  /  py tools/lan_test.py client      # 两个进程本机联机冒烟测试（bot 驱动）
```

`ClientSession(..., autoplay=True)` 让客户端由 bot 驱动，可作为「远程智能体」接入主机的模板。

## 代码结构

```
fpsrl/
  config.py  weapons.py        物理常量、武器参数（伤害、射速、后坐力图案、价格……）
  sim.py                       权威模拟：移动/碰撞、探头、射击、近战、投掷物、伤害、购买、事件
  geometry.py                  向量化射线-AABB、线段遮挡、球体遮挡
  modes/                       团队竞技、占点、回合制基类 + 爆破、彩六
  operators.py gadgets.py      彩六干员和装备（无人机、摄像头、路障、破门、干扰器……）
  destruction.py               可破坏 / 可加固墙体和地板
  maps/                        运输船、沙漠灰、占点小镇（含导入数据）、银行
  nav.py bots.py               分层导航图（跟随破坏和路障更新）、bot 感知 / 瞄准 / 模式目标 / 购买
  obs.py  envs.py              观测构造、Gymnasium / 多智能体环境
  render/                      Panda3D 渲染：场景、模型、贴图、特效、音效、第一人称手臂、HUD
    app.py menu.py             主菜单 / 暂停菜单 / 应用状态机
    game.py buymenu.py         本地对局、局域网主机、客户端会话；购买 / 干员菜单
    skins.py killbanner.py     程序化皮肤、击杀勋章
  net/                         局域网协议、主机（TCP + UDP 发现）、客户端
  profile.py                   玩家设置 / 皮肤存档
play.py                        游戏入口
tools/import_town.py           把厘米坐标的小镇点位数据转换成 fpsrl/maps/data/hp_town.json
tools/import_dust2.py          把 de_dust2 OBJ 转成渲染网格 + 碰撞盒（fpsrl/maps/data/dust2/）
tools/top_view.py              地图俯视图（地面高度、炸弹点、出生点、购买区）
```

沙漠灰模型：[CS 1.6 de_dust2](https://sketchfab.com/3d-models/de-dust2-cs-map-056008d59eb849a29c0ab6884c0c3d87) by vrchris，CC-BY 4.0。重新导入：`py tools/import_dust2.py de_dust2-cs-map.zip`。

新增地图：在 `fpsrl/maps/` 下返回一个 `MapData`（盒子、出生点、购买区、`objectives` 里的点位 / 炸弹点），并在 `maps/__init__.py` 的 `MAPS` 中注册支持的模式。新增模式：继承 `fpsrl/modes/base.py` 的 `GameMode`。新增武器：在 `weapons.py` 添加 `WeaponSpec`，在 `render/models.py` 添加模型构建函数。
