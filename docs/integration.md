# 接入开悟平台基线

本仓库只包含作者独立编写的文件。开悟平台 `hok1v1` 任务（V63.0.9）的基线框架版权归腾讯所有，没有收录；下面列出在基线上需要改动的位置，代码块只包含作者新增的部分，`...` 处为基线原有代码。

## 1. 放入掩码模块

把 `agent_ppo/feature/tactical_mask.py` 复制到基线的同名目录。模块只依赖 `numpy`。

## 2. 在观测处理中注入掩码（`agent_ppo/agent.py`）

掩码必须在采样之前写进 `legal_action`。这样训练采样（`predict`）、评估决策（`exploit`）、样本中记录的旧策略概率，以及学习端从样本里取出的掩码，都是同一份。

```python
from agent_ppo.feature.tactical_mask import apply_tactical_mask

# Agent.__init__ / reset 中：每局开始时清空局内记忆
self.tactical_memory = {}

def observation_process(self, observation):
    feature = ...                                    # 基线原有的特征处理
    legal_action = observation["legal_action"]
    if getattr(GameConfig, "USE_TACTICAL_MASK", False):
        legal_action = apply_tactical_mask(
            legal_action,
            observation.get("frame_state"),
            self.hero_camp,
            memory=self.tactical_memory,
            enable_movement=getattr(GameConfig, "USE_MOVEMENT_MASK", True),
            enable_flash_guidance=getattr(GameConfig, "USE_FLASH_GUIDANCE", True),
        )
    return ObsData(feature=feature, legal_action=legal_action, ...)
```

基线在构造样本时（`agent_ppo/feature/definition.py` 的 `build_frame`）读取的是环境原始的 `observation["legal_action"]`，需要改成读取已经过滤的掩码，否则学习端用的仍是未过滤的合法动作，新旧策略的支持集就不一致：

```python
raw_legal_action = (
    obs_data.legal_action if getattr(obs_data, "legal_action", None) is not None
    else observation["legal_action"]
)
legal_action = _update_legal_action(raw_legal_action, action)
```

之后选定按钮对应的 85 维掩码随样本保存，学习端用它对新策略做带掩码的 softmax，损失的掩码部分不需要再改。

## 3. 配置（`agent_ppo/conf/conf.py`）

```python
class GameConfig:
    USE_TACTICAL_MASK = True       # 消融开关：False 即“仅奖励”
    USE_MOVEMENT_MASK = True
    USE_FLASH_GUIDANCE = True
    REWARD_WEIGHT_DICT = {         # “仅掩码”只保留 tower_hp_point = 1.0，其余置 0
        "lane_progress": 0.25, "forward": 0.0,
        "tower_hp_point": 8.0, "crystal_hp_point": 10.0,
        "hp_point": 2.0, "kill": 1.0, "death": -1.2,
        "money": 0.06, "last_hit": 0.20, "exp": 0.20,
        "tower_shot": 0.25, "stall": 0.08,
    }

class Config:
    INIT_LEARNING_RATE_START = 5e-4
    TARGET_LR = 1e-4
    TARGET_STEP = 15000
    BETA_START = 0.015
    ENT_HEAD_SCALE = (0.20, 1.0, 1.0, 1.0, 1.0, 0.20)   # 按钮头与目标头的熵缩放
    USE_COLD_START_PRIOR = True
```

## 4. 冷启动偏置与分支熵缩放（`agent_ppo/model/model.py`）

在模型构造函数末尾，给按钮头输出层的偏置加上攻击与技能的先验：

```python
if getattr(Config, "USE_COLD_START_PRIOR", False):
    btn_layer = self.label_mlp["hero_label0_mlp"].fc_layers[-1]
    with torch.no_grad():
        btn_layer.bias[3] += 0.85  # 普攻
        btn_layer.bias[4] += 0.55  # 一技能
        btn_layer.bias[5] += 0.60  # 二技能
        btn_layer.bias[6] += 0.55  # 三技能
```

在熵损失按分支累加的循环中乘上缩放系数：

```python
head_scale = 1.0
if hasattr(Config, "ENT_HEAD_SCALE") and len(Config.ENT_HEAD_SCALE) > task_index:
    head_scale = float(Config.ENT_HEAD_SCALE[task_index])
entropy_loss_list.append(head_scale * torch.sum(negative_entropy * branch_weight) / branch_weight_sum)
```

## 5. 混合塑形奖励（`agent_ppo/feature/reward_process.py`）

奖励按“差分项”（双方同一状态量的帧间变化之差）与“单边项”（只看己方）两类合成，各项定义与权重见技术报告表 4。`get_reward` 中单边项直接取己方当帧值：

```python
if reward_name in ("forward", "last_hit", "lane_progress", "tower_shot", "stall"):
    reward_struct.value = own.cur_frame_value
else:
    reward_struct.value = (own.cur_frame_value - enemy.cur_frame_value
                           - own.last_frame_value + enemy.last_frame_value)
```

在逐项计算 `cur_frame_value` 的分支中新增的各项：

```python
elif reward_name == "lane_progress":
    self._update_lane_progress(reward_struct, main_hero, main_target, enemy_target,
                               enemy_tower, npc_list, camp)
elif reward_name == "crystal_hp_point":
    if main_spring and main_spring["max_hp"] > 0:
        reward_struct.cur_frame_value = main_spring["hp"] / main_spring["max_hp"]
elif reward_name == "hp_point":                        # 复活首帧重置，避免把复活算成回血
    was_dead = getattr(reward_struct, "was_dead", False)
    reward_struct.cur_frame_value = hero_hp_rate
    if was_dead and main_hero_hp > 0:
        reward_struct.last_frame_value = hero_hp_rate
    reward_struct.was_dead = main_hero_hp <= 0
elif reward_name == "money":
    reward_struct.cur_frame_value = self._gold_income(reward_struct, main_hero) / 100.0
elif reward_name == "last_hit":
    reward_struct.cur_frame_value = self._compute_last_hit(reward_struct, main_hero, enemy_tower)
elif reward_name == "exp":
    level = main_hero.get("level")
    if level is not None:
        exp_frac = min(0.99, float(main_hero.get("exp") or 0.0) / 1500.0)
        reward_struct.cur_frame_value = max(reward_struct.cur_frame_value, float(level) + exp_frac)
elif reward_name == "tower_shot":                      # 敌塔锁定自己且掉血时计 -1
    cur_hp = main_hero.get("hp", 0)
    prev_hp = getattr(reward_struct, "prev_hp", cur_hp)
    was_dead = getattr(reward_struct, "was_dead", False)
    hp_loss = max(0, prev_hp - cur_hp) if not was_dead and cur_hp > 0 else 0
    targeting_me = bool(enemy_tower and main_hero
                        and enemy_tower.get("attack_target") == main_hero.get("runtime_id"))
    reward_struct.cur_frame_value = -1.0 if (targeting_me and hp_loss > 0) else 0.0
    reward_struct.prev_hp = cur_hp
    reward_struct.was_dead = cur_hp <= 0
elif reward_name == "stall":                           # 满血、远离目标且附近无敌军时计 -1
    ...
```

其中三个辅助函数：

```python
def _update_lane_progress(self, reward_struct, main_hero, main_target, enemy_target,
                          enemy_tower, npc_list, camp):
    if not main_target or not enemy_target or main_hero.get("hp", 0) <= 0:
        reward_struct.was_inactive = True
        reward_struct.in_tower_danger = False
        reward_struct.cur_frame_value = 0.0
        return
    main_pos = (main_target["location"]["x"], main_target["location"]["z"])
    enemy_pos = (enemy_target["location"]["x"], enemy_target["location"]["z"])
    hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
    dist_main2emy = math.dist(main_pos, enemy_pos)
    if dist_main2emy <= 0:
        reward_struct.cur_frame_value = 0.0
        return
    prog = max(-1.0, min(0.8, (dist_main2emy - math.dist(hero_pos, enemy_pos)) / dist_main2emy))

    max_hp = main_hero.get("max_hp", 0)
    hp_rate = main_hero.get("hp", 0) / max_hp if max_hp > 0 else 0.0
    tower_danger = False                               # 无己方小兵时进入敌塔范围
    if enemy_tower and enemy_tower.get("hp", 0) > 0:
        et_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        if math.dist(hero_pos, et_pos) < 8500.0:
            tower_danger = not any(
                npc.get("camp") == camp and npc.get("sub_type") not in (21, 22, 23, 24)
                and npc.get("hp", 0) > 0
                and math.dist((npc["location"]["x"], npc["location"]["z"]), et_pos) < 8500.0
                for npc in npc_list)

    if getattr(reward_struct, "was_inactive", True):
        reward_struct.prev_prog = prog
        reward_struct.in_tower_danger = tower_danger
        reward_struct.was_inactive = False
        reward_struct.cur_frame_value = 0.0
        return
    delta = prog - reward_struct.prev_prog
    reward_struct.prev_prog = prog
    was_danger = getattr(reward_struct, "in_tower_danger", False)
    reward_struct.in_tower_danger = tower_danger
    if tower_danger or was_danger:
        reward_struct.cur_frame_value = -delta         # 塔下无兵时往里走扣分、退出来加分
    elif hp_rate < 0.45 and delta < 0.0:
        reward_struct.cur_frame_value = 0.0            # 低血撤退不扣分
    else:
        reward_struct.cur_frame_value = delta

@staticmethod
def _gold_income(reward_struct, hero):
    """V63 云端帧只给当前金币 money（买装备会下降），因此累加其增量作为总收入。"""
    if "money_cnt" in hero:
        return float(hero["money_cnt"])
    gold = hero.get("money")
    income = getattr(reward_struct, "gold_income", 0.0)
    if gold is None:
        return income
    prev = getattr(reward_struct, "prev_gold", None)
    if prev is not None and gold > prev:
        income += gold - prev
    reward_struct.prev_gold = gold
    reward_struct.gold_income = income
    return income

def _compute_last_hit(self, reward_struct, main_hero, enemy_tower):
    """非击杀、非破塔帧的金币增量 >= 55 计 1，>= 38 计 0.5。"""
    money_cnt = main_hero.get("money_cnt", main_hero.get("money"))
    kill_cnt = main_hero.get("kill_cnt")
    tower_hp = enemy_tower.get("hp") if enemy_tower else None
    prev_money = getattr(reward_struct, "prev_money_cnt", None)
    prev_kills = getattr(reward_struct, "prev_kill_cnt", None)
    prev_tower_hp = getattr(reward_struct, "prev_tower_hp", None)
    val = 0.0
    if prev_money is not None and money_cnt is not None:
        dm = float(money_cnt) - float(prev_money)
        dk = (int(kill_cnt) - int(prev_kills)) if (prev_kills is not None and kill_cnt is not None) else 0
        tower_fell = (prev_tower_hp is not None and prev_tower_hp > 0
                      and tower_hp is not None and tower_hp <= 0)
        if dk == 0 and not tower_fell:
            val = 1.0 if dm >= 55.0 else 0.5 if dm >= 38.0 else 0.0
    if money_cnt is not None:
        reward_struct.prev_money_cnt = money_cnt
    if kill_cnt is not None:
        reward_struct.prev_kill_cnt = kill_cnt
    if tower_hp is not None:
        reward_struct.prev_tower_hp = tower_hp
    return val
```
