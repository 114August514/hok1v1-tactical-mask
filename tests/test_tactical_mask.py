import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from agent_ppo.feature.tactical_mask import (
    BUTTON_ATTACK,
    BUTTON_CHOSEN,
    BUTTON_HEAL,
    BUTTON_MOVE,
    BUTTON_NONE_0,
    BUTTON_NONE_1,
    BUTTON_RECALL,
    BUTTON_SKILL1,
    BUTTON_SKILL2,
    BUTTON_SKILL3,
    EXPECTED_LEGAL_LEN,
    LABEL_SIZES,
    TARGET_ENEMY_HERO,
    TARGET_SELF,
    TARGET_TOWER,
    apply_tactical_mask,
)


def _legal_vec():
    legal = np.zeros(EXPECTED_LEGAL_LEN, dtype=np.float32)
    legal[: LABEL_SIZES[0]] = 1.0
    offset = LABEL_SIZES[0]
    for size in LABEL_SIZES[1:-1]:
        legal[offset : offset + size] = 1.0
        offset += size
    legal[offset:] = 1.0
    return legal


def _frame(hero_x=0, hero_hp=100, enemy_x=4000, enemy_hp=100, tower_x=20000, minions=()):
    npc_states = [
        {"camp": 1, "sub_type": 21, "hp": 100, "max_hp": 100, "location": {"x": -10000, "z": 0}},
        {"camp": 2, "sub_type": 21, "hp": 100, "max_hp": 100, "location": {"x": tower_x, "z": 0}},
    ]
    for mx in minions:
        npc_states.append({"camp": 1, "sub_type": 1, "hp": 50, "max_hp": 50, "location": {"x": mx, "z": 0}})
    return {
        "hero_states": [
            {"camp": 1, "hp": hero_hp, "max_hp": 100, "location": {"x": hero_x, "z": 0}},
            {"camp": 2, "hp": enemy_hp, "max_hp": 100, "location": {"x": enemy_x, "z": 0}},
        ],
        "npc_states": npc_states,
    }


def test_mask_drops_idle_recall_and_gates_heal():
    legal = _legal_vec()
    # Full HP hero with enemy in range: drop None0/None1/Recall and Heal
    masked = apply_tactical_mask(legal, _frame(hero_hp=95, enemy_x=4000), main_camp=1)
    assert masked[BUTTON_NONE_0] == 0.0
    assert masked[BUTTON_NONE_1] == 0.0
    assert masked[BUTTON_RECALL] == 0.0
    assert masked[BUTTON_HEAL] == 0.0
    assert masked[BUTTON_MOVE] == 1.0
    assert masked[BUTTON_ATTACK] == 1.0

    # Injured hero below 85% HP keeps Heal
    masked_low = apply_tactical_mask(legal, _frame(hero_hp=60, enemy_x=4000), main_camp=1)
    assert masked_low[BUTTON_HEAL] == 1.0


def test_mask_gates_air_casts_and_ignores_fog_sentinel():
    legal = _legal_vec()
    # Nearest target at 12000: Attack(7500)/Skill1(8500)/Skill3(8500) blocked, Skill2(15000) allowed
    masked = apply_tactical_mask(legal, _frame(enemy_x=12000, tower_x=30000), main_camp=1)
    assert masked[BUTTON_ATTACK] == 0.0
    assert masked[BUTTON_SKILL1] == 0.0
    assert masked[BUTTON_SKILL3] == 0.0
    assert masked[BUTTON_SKILL2] == 1.0

    # Fog-of-war sentinel coordinates (100000) mean no target in sight -> block all offense
    fog_frame = _frame(enemy_x=100000, tower_x=100000)
    masked_fog = apply_tactical_mask(legal, fog_frame, main_camp=1)
    for b in (BUTTON_ATTACK, BUTTON_SKILL1, BUTTON_SKILL2, BUTTON_SKILL3):
        assert masked_fog[b] == 0.0
    assert masked_fog[BUTTON_MOVE] == 1.0


def test_mask_prevents_bare_tower_dive_and_allows_with_minions():
    legal = _legal_vec()
    # Inside 8500 of enemy tower (at 20000), hero at 14000, no friendly minions
    bare_dive = _frame(hero_x=14000, enemy_x=22000, tower_x=20000, minions=())
    masked_bare = apply_tactical_mask(legal, bare_dive, main_camp=1)
    assert masked_bare[BUTTON_ATTACK] == 0.0
    assert masked_bare[BUTTON_SKILL1] == 0.0
    assert masked_bare[BUTTON_MOVE] == 1.0

    target_offset = sum(LABEL_SIZES[:-1])
    tmat = masked_bare[target_offset:].reshape(LABEL_SIZES[0], LABEL_SIZES[-1])
    assert tmat[BUTTON_ATTACK, TARGET_TOWER] == 0.0
    assert tmat[BUTTON_ATTACK, TARGET_ENEMY_HERO] == 0.0

    # With a friendly minion tanking under tower, offense is allowed
    shielded_push = _frame(hero_x=14000, enemy_x=18000, tower_x=20000, minions=(16000,))
    masked_ok = apply_tactical_mask(legal, shielded_push, main_camp=1)
    assert masked_ok[BUTTON_ATTACK] == 1.0


def test_mask_disengages_when_critically_low_and_keeps_fallback():
    legal = _legal_vec()
    # Hero below 25% HP near healthy enemy hero -> mask offense to force retreat
    danger = _frame(hero_x=0, hero_hp=20, enemy_x=3000, enemy_hp=80)
    masked = apply_tactical_mask(legal, danger, main_camp=1)
    assert masked[BUTTON_ATTACK] == 0.0
    assert masked[BUTTON_MOVE] == 1.0

    # If only Attack is legal in raw vector and mask would zero everything out, fallback restores raw
    attack_only = np.zeros(EXPECTED_LEGAL_LEN, dtype=np.float32)
    attack_only[BUTTON_ATTACK] = 1.0
    attack_only[sum(LABEL_SIZES[:-1]) :] = 1.0
    fallback = apply_tactical_mask(attack_only, _frame(enemy_x=30000, tower_x=30000), main_camp=1)
    assert fallback[BUTTON_ATTACK] == 1.0


def test_build_frame_and_update_legal_action_preserve_masked_obs_data(monkeypatch):
    def _create_cls(name, **fields):
        return lambda **kw: SimpleNamespace(**{**fields, **kw})

    stub_func = ModuleType("common_python.utils.common_func")
    stub_func.create_cls = _create_cls
    stub_func.Frame = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "common_python", ModuleType("common_python"))
    monkeypatch.setitem(sys.modules, "common_python.utils", ModuleType("common_python.utils"))
    monkeypatch.setitem(sys.modules, "common_python.utils.common_func", stub_func)
    # 需要开悟平台基线中的 agent_ppo.feature.definition；只有掩码模块时跳过
    defn = pytest.importorskip("agent_ppo.feature.definition")

    raw_legal = _legal_vec()
    masked = apply_tactical_mask(raw_legal, _frame(enemy_x=30000, tower_x=30000), main_camp=1)
    obs_data = SimpleNamespace(feature=np.zeros(350, dtype=np.float32), legal_action=masked)
    act_data = SimpleNamespace(
        prob=np.zeros(sum(LABEL_SIZES), dtype=np.float32),
        value=np.array([0.0], dtype=np.float32),
        action=np.array([BUTTON_MOVE, 0, 0, 0, 0, 0]),
        lstm_cell=np.zeros(512, dtype=np.float32),
        lstm_hidden=np.zeros(512, dtype=np.float32),
    )
    fake_agent = SimpleNamespace(hero_camp=1, obs_data=obs_data, act_data=act_data)
    observation = {
        "frame_state": {"frame_no": 1, "hero_states": [{"camp": 1, "hp": 100}]},
        "reward": {"reward_sum": 0.0},
        "sub_action_mask": {str(BUTTON_MOVE): np.ones(6, dtype=np.float32)},
        "legal_action": raw_legal,
    }
    frame_obj = defn.build_frame(fake_agent, observation)
    # Recorded legal_action comes from masked obs_data.legal_action, not raw_legal
    assert frame_obj.legal_action[BUTTON_NONE_0] == 0.0
    assert frame_obj.legal_action[BUTTON_RECALL] == 0.0
    assert frame_obj.legal_action[BUTTON_ATTACK] == 0.0


def test_flash_escape_execute_and_idle_blocking():
    legal = _legal_vec()

    # 1. Idle state: healthy, no danger -> Flash (Button 8) is zeroed to prevent burn
    safe_state = _frame(hero_x=0, hero_hp=100, enemy_x=20000, tower_x=30000)
    masked_safe = apply_tactical_mask(legal, safe_state, main_camp=1)
    assert masked_safe[BUTTON_CHOSEN] == 0.0

    # 2. Escape Flash: under bare tower danger with hp < 55%
    tower_escape = _frame(hero_x=14000, hero_hp=40, enemy_x=25000, tower_x=20000, minions=())
    masked_esc = apply_tactical_mask(legal, tower_escape, main_camp=1)
    assert masked_esc[BUTTON_CHOSEN] == 1.0
    # For Camp 1 at (14000, 0), home is at (-20000, -20000), dx < 0, dz < 0 -> skill_x and skill_z < 8
    target_offset = sum(LABEL_SIZES[:-1])
    btn8_tgt = masked_esc[target_offset + BUTTON_CHOSEN * LABEL_SIZES[-1] : target_offset + (BUTTON_CHOSEN + 1) * LABEL_SIZES[-1]]
    assert btn8_tgt[TARGET_SELF] == 1.0
    skill_x_bin = int(np.argmax(masked_esc[44:60]))
    skill_z_bin = int(np.argmax(masked_esc[60:76]))
    assert skill_x_bin < 8
    assert skill_z_bin < 8

    # 3. Execute Flash: healthy hero (hp=60%), enemy hero low (ehp=20%) at distance 8000
    execute_state = _frame(hero_x=0, hero_hp=60, enemy_x=8000, enemy_hp=20, tower_x=30000)
    masked_exec = apply_tactical_mask(legal, execute_state, main_camp=1)
    assert masked_exec[BUTTON_CHOSEN] == 1.0
    # Direction toward enemy at (+8000, 0) -> skill_x > 8, skill_z == 8
    assert int(np.argmax(masked_exec[44:60])) > 8


def test_recovery_and_corridor_movement_locking():
    legal = _legal_vec()

    # Low HP near base -> retreat to fountain, lock Move X/Z toward fountain
    retreat_state = _frame(hero_x=-15000, hero_hp=30, enemy_x=10000, tower_x=20000)
    masked_ret = apply_tactical_mask(legal, retreat_state, main_camp=1)
    # Move bins toward (-20000, -20000): dx < 0, dz < 0 -> bin < 8
    move_x_bin = int(np.argmax(masked_ret[12:28]))
    move_z_bin = int(np.argmax(masked_ret[28:44]))
    assert move_x_bin < 8
    assert move_z_bin < 8

    # Outbound corridor: healthy hero at 0, no minions, no enemy -> lock Move X/Z toward enemy tower at (20000, 0)
    corridor_state = _frame(hero_x=0, hero_hp=90, enemy_x=25000, tower_x=20000)
    masked_corridor = apply_tactical_mask(legal, corridor_state, main_camp=1)
    cx_bin = int(np.argmax(masked_corridor[12:28]))
    cz_bin = int(np.argmax(masked_corridor[28:44]))
    assert cx_bin > 8  # Moving forward along +X


def test_dead_hero_action_masking():
    legal = _legal_vec()
    dead_state = _frame(hero_x=0, hero_hp=0, enemy_x=5000, tower_x=20000)
    masked = apply_tactical_mask(legal, dead_state, main_camp=1)
    # All combat and utility buttons zeroed
    for b in (BUTTON_ATTACK, BUTTON_SKILL1, BUTTON_SKILL2, BUTTON_SKILL3, BUTTON_HEAL, BUTTON_CHOSEN):
        assert masked[b] == 0.0


def test_camp2_directional_symmetry():
    legal = _legal_vec()
    # Real V63 GameCore negates (move_x, move_z) and (skill_x, skill_z) for Camp 2.
    # Camp 2 hero at (10000, 0) pushing toward Camp 1 enemy tower at (-10000, 0):
    # world dx = -20000 < 0 -> mirrored action unit ux = +1.0 -> move_x bin > 8
    c2_push = {
        "hero_states": [
            {"camp": 2, "hp": 100, "max_hp": 100, "location": {"x": 10000, "z": 0}},
            {"camp": 1, "hp": 100, "max_hp": 100, "location": {"x": -25000, "z": 0}},
        ],
        "npc_states": [
            {"camp": 2, "sub_type": 21, "hp": 100, "max_hp": 100, "location": {"x": 20000, "z": 0}},
            {"camp": 2, "sub_type": 23, "hp": 6000, "max_hp": 6000, "location": {"x": 45785, "z": 0}},
            {"camp": 1, "sub_type": 21, "hp": 100, "max_hp": 100, "location": {"x": -10000, "z": 0}},
        ],
    }
    masked_push = apply_tactical_mask(legal, c2_push, main_camp=2)
    assert int(np.argmax(masked_push[12:28])) > 8

    # Camp 2 low-HP fountain retreat from (10000, 0) toward own fountain at (45785, 0):
    # world dx = +35785 > 0 -> mirrored action unit ux = -1.0 -> move_x bin < 8
    c2_retreat = {
        "hero_states": [
            {"camp": 2, "hp": 25, "max_hp": 100, "location": {"x": 10000, "z": 0}},
            {"camp": 1, "hp": 100, "max_hp": 100, "location": {"x": -25000, "z": 0}},
        ],
        "npc_states": c2_push["npc_states"],
    }
    masked_retreat = apply_tactical_mask(legal, c2_retreat, main_camp=2)
    assert int(np.argmax(masked_retreat[12:28])) < 8

    # Camp 2 Flash escape toward home (45785, 0): world dx > 0 -> mirrored skill_x bin < 8
    c2_flash_esc = {
        "hero_states": [
            {"camp": 2, "hp": 15, "max_hp": 100, "location": {"x": 10000, "z": 0}},
            {"camp": 1, "hp": 80, "max_hp": 100, "location": {"x": 8500, "z": 0}},
        ],
        "npc_states": c2_push["npc_states"],
    }
    masked_esc = apply_tactical_mask(legal, c2_flash_esc, main_camp=2)
    assert masked_esc[BUTTON_CHOSEN] == 1.0
    assert int(np.argmax(masked_esc[44:60])) < 8

    # Camp 2 Flash execute toward Camp 1 enemy at (-8000, 0) from (0, 0):
    # world dx = -8000 < 0 -> mirrored skill_x bin > 8
    c2_flash_exec = {
        "hero_states": [
            {"camp": 2, "hp": 70, "max_hp": 100, "location": {"x": 0, "z": 0}},
            {"camp": 1, "hp": 15, "max_hp": 100, "location": {"x": -8000, "z": 0}},
        ],
        "npc_states": [
            {"camp": 2, "sub_type": 21, "hp": 100, "max_hp": 100, "location": {"x": 20000, "z": 0}},
            {"camp": 2, "sub_type": 23, "hp": 6000, "max_hp": 6000, "location": {"x": 45785, "z": 0}},
            {"camp": 1, "sub_type": 21, "hp": 100, "max_hp": 100, "location": {"x": -30000, "z": 0}},
        ],
    }
    masked_exec = apply_tactical_mask(legal, c2_flash_exec, main_camp=2)
    assert masked_exec[BUTTON_CHOSEN] == 1.0
    assert int(np.argmax(masked_exec[44:60])) > 8


def test_fountain_subtype23_vs_crystal_subtype24():
    legal = _legal_vec()
    # Injured Camp 1 hero standing RIGHT AT own Crystal (-31310, -31120) must continue
    # moving (-X, -Z) toward Fountain (sub_type=23 at (-45396, -45467)), NOT stop at Crystal.
    state = {
        "hero_states": [
            {"camp": 1, "hp": 30, "max_hp": 100, "location": {"x": -31310, "z": -31120}},
            {"camp": 2, "hp": 100, "max_hp": 100, "location": {"x": 25000, "z": 25000}},
        ],
        "npc_states": [
            {"camp": 1, "sub_type": 24, "hp": 9000, "max_hp": 9000, "location": {"x": -31310, "z": -31120}},
            {"camp": 1, "sub_type": 23, "hp": 6000, "max_hp": 6000, "location": {"x": -45396, "z": -45467}},
            {"camp": 2, "sub_type": 21, "hp": 12000, "max_hp": 12000, "location": {"x": 13131, "z": 13258}},
        ],
    }
    masked = apply_tactical_mask(legal, state, main_camp=1)
    assert int(np.argmax(masked[12:28])) < 8
    assert int(np.argmax(masked[28:44])) < 8


def test_dead_tower_switches_corridor_target_to_crystal():
    legal = _legal_vec()
    mem = {}
    # Step 1: Enemy outer tower at (13131, 0) alive, crystal at (31250, 0)
    alive_state = {
        "hero_states": [
            {"camp": 1, "hp": 90, "max_hp": 100, "location": {"x": 13000, "z": 0}},
            {"camp": 2, "hp": 100, "max_hp": 100, "location": {"x": 35000, "z": 0}},
        ],
        "npc_states": [
            {"camp": 2, "sub_type": 21, "hp": 12000, "max_hp": 12000, "location": {"x": 13131, "z": 0}},
            {"camp": 2, "sub_type": 24, "hp": 9000, "max_hp": 9000, "location": {"x": 31250, "z": 0}},
        ],
    }
    apply_tactical_mask(legal, alive_state, main_camp=1, memory=mem)
    assert mem["enemy_tower_hp"] == 12000.0

    # Step 2: Tower destroyed (hp == 0); hero past old tower at (15000, 0) must push +X toward crystal
    dead_tower_state = {
        "hero_states": [
            {"camp": 1, "hp": 90, "max_hp": 100, "location": {"x": 15000, "z": 0}},
            {"camp": 2, "hp": 100, "max_hp": 100, "location": {"x": 35000, "z": 0}},
        ],
        "npc_states": [
            {"camp": 2, "sub_type": 21, "hp": 0, "max_hp": 12000, "location": {"x": 13131, "z": 0}},
            {"camp": 2, "sub_type": 24, "hp": 9000, "max_hp": 9000, "location": {"x": 31250, "z": 0}},
        ],
    }
    masked = apply_tactical_mask(legal, dead_tower_state, main_camp=1, memory=mem)
    assert mem["enemy_tower_hp"] == 0.0
    assert int(np.argmax(masked[12:28])) > 8


def _step(frame, mem):
    trace = {}
    masked = apply_tactical_mask(_legal_vec(), frame, main_camp=1, memory=mem, trace=trace)
    return masked, trace


def _target_row(masked, button):
    offset, width = sum(LABEL_SIZES[:-1]), LABEL_SIZES[-1]
    return masked[offset + button * width : offset + (button + 1) * width]


def test_fountain_trip_is_kept_until_healed_and_cleared_by_death():
    mem = {}
    _, trace = _step(_frame(hero_hp=30, enemy_x=25000), mem)
    assert trace["rec"] == "fountain"

    # Halfway home at 50%: a fresh decision would push the corridor, the trip keeps going home.
    _, fresh = _step(_frame(hero_hp=50, enemy_x=25000), {})
    assert fresh["corridor"]
    masked, trace = _step(_frame(hero_hp=50, enemy_x=25000), mem)
    assert trace["rec"] == "fountain" and not trace["corridor"]
    assert int(np.argmax(masked[12:28])) < 8

    _, trace = _step(_frame(hero_hp=80, enemy_x=25000), mem)
    assert trace["rec"] is None and trace["corridor"]

    _step(_frame(hero_hp=30, enemy_x=25000), mem)
    _step(_frame(hero_hp=0, enemy_x=25000), mem)
    _, trace = _step(_frame(hero_hp=50, enemy_x=25000), mem)
    assert trace["rec"] is None


def test_urgent_tower_ends_trip_at_65_percent():
    def urgent(hp):
        frame = _frame(hero_hp=hp, enemy_x=25000)
        frame["npc_states"][0]["hp"] = 30  # own tower below TOWER_URGENT_HP_RATIO
        return frame

    mem = {}
    _step(_frame(hero_hp=30, enemy_x=25000), mem)
    assert _step(urgent(60), mem)[1]["rec"] == "fountain"
    assert _step(urgent(65), mem)[1]["rec"] is None


def test_tower_danger_escape_does_not_start_a_trip():
    mem = {}
    _, trace = _step(_frame(hero_x=15000, hero_hp=90, enemy_x=40000, tower_x=20000), mem)
    assert trace["tower_danger"] and trace["rec"] == "fountain"
    _, trace = _step(_frame(hero_x=0, hero_hp=90, enemy_x=40000, tower_x=20000), mem)
    assert trace["rec"] is None


def test_no_execute_detour_while_going_home():
    low_enemy_near = _frame(hero_hp=30, enemy_x=5000, enemy_hp=10)
    assert _step(low_enemy_near, {})[1]["rec"] is None  # fresh: execute exception applies
    mem = {}
    _step(_frame(hero_hp=30, enemy_x=25000), mem)
    assert _step(low_enemy_near, mem)[1]["rec"] == "fountain"


def test_corridor_needs_half_hp():
    assert not _step(_frame(hero_hp=45, enemy_x=25000), {})[1]["corridor"]
    assert _step(_frame(hero_hp=55, enemy_x=25000), {})[1]["corridor"]


def test_hero_pressure_restricts_targets_or_attack():
    def run(enemy_x, hps):
        mem = {}
        for hp in hps:
            frame = _frame(hero_hp=hp, enemy_x=enemy_x)
            frame["npc_states"].append({"camp": 2, "sub_type": 1, "hp": 50, "max_hp": 50, "location": {"x": 3000, "z": 0}})
            masked, trace = _step(frame, mem)
        return masked, trace

    # Not enough history yet: no pressure
    assert not run(5000, (90, 85, 80))[1]["hero_pressure"]

    # Enemy within attack range while we lose HP: only the hero (or no target) remains
    masked, trace = run(5000, (90, 85, 80, 75))
    assert trace["hero_pressure"]
    assert masked[BUTTON_ATTACK] == 1.0
    row = _target_row(masked, BUTTON_ATTACK)
    assert row[TARGET_ENEMY_HERO] == 1.0
    assert row[3:9].sum() == 0.0  # soldiers, tower, monster

    # Enemy beyond our attack range: stop attacking (a minion is in range, so range gating alone would allow it)
    masked, trace = run(7800, (90, 85, 80, 75))
    assert trace["hero_pressure"] and masked[BUTTON_ATTACK] == 0.0

    # Steady HP next to the enemy is farming, not pressure: minion targets stay
    masked, trace = run(5000, (90, 90, 90, 90))
    assert not trace["hero_pressure"]
    assert _target_row(masked, BUTTON_ATTACK)[3:7].sum() == 4.0
