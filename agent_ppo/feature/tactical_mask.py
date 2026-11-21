#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Tactical action-mask refinement for V63 HoK1v1 (Luban No. 7 adaptation).

Applied consistently in Agent.observation_process and recorded via ObsData so
predict(), exploit(), old_prob, and Learner compute_loss observe the exact same
legal action masks.

Includes:
- Button air-cast gating for Attack, Skill 1, 2, 3
- Losing-duel / tower-danger combat retreat masking
- Recovery movement (tower cake detour & fountain heal)
- Safe outbound corridor marching toward enemy tower/crystal
- Flash (80115, Button 8) tactical gating & directional guidance (escape / execute)
- Fog memory fallback for map landmarks
"""

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np

BUTTON_NONE_0 = 0
BUTTON_NONE_1 = 1
BUTTON_MOVE = 2
BUTTON_ATTACK = 3
BUTTON_SKILL1 = 4
BUTTON_SKILL2 = 5
BUTTON_SKILL3 = 6
BUTTON_HEAL = 7
BUTTON_CHOSEN = 8  # Summoner skill: Flash (80115) in cloud V63
BUTTON_RECALL = 9
BUTTON_SKILL4 = 10
BUTTON_EQUIP = 11

COMBAT_BUTTONS = (BUTTON_MOVE, BUTTON_ATTACK, BUTTON_SKILL1, BUTTON_SKILL2, BUTTON_SKILL3)
OFFENSE_BUTTONS = (BUTTON_ATTACK, BUTTON_SKILL1, BUTTON_SKILL2, BUTTON_SKILL3)

TARGET_NONE = 0
TARGET_ENEMY_HERO = 1
TARGET_SELF = 2
TARGET_SOLDIER_0 = 3
TARGET_SOLDIER_1 = 4
TARGET_SOLDIER_2 = 5
TARGET_SOLDIER_3 = 6
TARGET_TOWER = 7
TARGET_MONSTER = 8
NON_HERO_TARGETS = [TARGET_SOLDIER_0, TARGET_SOLDIER_1, TARGET_SOLDIER_2, TARGET_SOLDIER_3, TARGET_TOWER, TARGET_MONSTER]

# Radii and tactical thresholds
ATTACK_RADIUS = 7500.0
SKILL_RADIUS = 8500.0
SKILL2_RADIUS = 15000.0
ORGAN_RADIUS = 1500.0
TOWER_DANGER_RANGE = 8500.0
HEAL_HP_THRESHOLD = 0.85

# Combat retreat thresholds
DISENGAGE_HP = 0.40
DISENGAGE_HP_GAP = 0.10
EXECUTE_HP = 0.28
MIN_EXECUTE_HP = 0.16
RETREAT_HERO_DIST = 9000.0

# Hero pressure: a visible enemy hero this close while our HP fell over the last
# HERO_PRESSURE_STEPS steps means we are being traded on, not farming.
HERO_PRESSURE_DIST = 8000.0
HERO_PRESSURE_STEPS = 3

# Recovery thresholds
HOME_HP = 0.40
CRITICAL_HOME_HP = 0.24
CAKE_HP = 0.55
CAKE_MAX_DETOUR = 12000.0
HOME_DONE_HP = 0.80
HOME_URGENT_DONE_HP = 0.65
TOWER_URGENT_HP_RATIO = 0.35
HOME_BASE_DIST = 12000.0
FOUNTAIN_STAND_DIST = 1200.0

# Outbound corridor thresholds
CORRIDOR_HERO_DIST = 9500.0
CORRIDOR_MINION_DIST = 8000.0
CORRIDOR_MIN_HP = 0.50
CORRIDOR_WAIT_TOWER_DIST = 9200.0
CORRIDOR_PUSH_TOWER_DIST = 8000.0

# Flash guidance constants
FLASH_ESCAPE_HERO_DIST = 6000.0
FLASH_EXECUTE_MIN_DIST = 6800.0
FLASH_EXECUTE_MAX_DIST = 11500.0
FLASH_EXECUTE_ENEMY_HP = 0.25

SENTINEL_COORD = 100000
ORGAN_SUBTYPES = (21, 22, 23, 24)
LABEL_SIZES = (12, 16, 16, 16, 16, 9)
EXPECTED_LEGAL_LEN = sum(LABEL_SIZES[:-1]) + LABEL_SIZES[0] * LABEL_SIZES[-1]  # 184


def _unit_xy(unit: Dict[str, Any]) -> Tuple[float, float]:
    loc = unit.get("location") or {}
    return float(loc.get("x", 0.0)), float(loc.get("z", 0.0))


def _dist(pos_a: Tuple[float, float], pos_b: Tuple[float, float]) -> float:
    return math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])


def _is_sentinel(pos: Tuple[float, float]) -> bool:
    return abs(pos[0]) >= SENTINEL_COORD and abs(pos[1]) >= SENTINEL_COORD


def _default_landmarks(camp: Any) -> Dict[str, Tuple[float, float]]:
    if camp in (2, "PLAYERCAMP_2"):
        return {
            "home": (45785.0, 45856.0),
            "main_tower": (13131.0, 13258.0),
            "enemy_tower": (-13138.0, -12940.0),
            "enemy_crystal": (-31310.0, -31120.0),
        }
    return {
        "home": (-45396.0, -45467.0),
        "main_tower": (-13138.0, -12940.0),
        "enemy_tower": (13131.0, 13258.0),
        "enemy_crystal": (31250.0, 31350.0),
    }


def _extract_tactical_state(
    frame_state: Optional[Dict[str, Any]],
    main_camp: int,
    memory: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(frame_state, dict):
        return None

    main_hero = None
    enemy_hero = None
    for hero in frame_state.get("hero_states") or []:
        if hero.get("camp") == main_camp:
            main_hero = hero
        else:
            enemy_hero = hero
    if main_hero is None:
        return None

    main_hp = float(main_hero.get("hp", 0.0))
    main_max_hp = float(main_hero.get("max_hp", 0.0))
    main_hp_ratio = main_hp / main_max_hp if main_max_hp > 0.0 else 0.0
    main_pos = _unit_xy(main_hero)
    main_rtid = main_hero.get("runtime_id")

    enemy_hero_dist = float("inf")
    enemy_hp_ratio = 1.0
    enemy_pos = None
    enemy_visible = False
    if enemy_hero is not None:
        ehp = float(enemy_hero.get("hp", 0.0))
        emax = float(enemy_hero.get("max_hp", 0.0))
        enemy_hp_ratio = ehp / emax if emax > 0.0 else 1.0
        epos = _unit_xy(enemy_hero)
        if ehp > 0.0 and not _is_sentinel(epos):
            enemy_hero_dist = _dist(main_pos, epos)
            enemy_pos = epos
            enemy_visible = True

    landmarks = _default_landmarks(main_camp)
    main_tower = None
    main_spring = None
    enemy_tower = None
    enemy_crystal = None
    saw_dead_main_tower = False
    saw_dead_enemy_tower = False
    enemy_minions = []
    friendly_minions = []

    for npc in frame_state.get("npc_states") or []:
        hp_val = float(npc.get("hp", 0.0))
        subtype = npc.get("sub_type")
        npc_camp = npc.get("camp")
        if subtype == 21:
            if hp_val <= 0.0:
                if npc_camp == main_camp:
                    saw_dead_main_tower = True
                else:
                    saw_dead_enemy_tower = True
                continue
        elif hp_val <= 0.0:
            continue

        if npc_camp == main_camp:
            if subtype == 21:
                main_tower = npc
            elif subtype == 23:
                main_spring = npc
            elif subtype not in ORGAN_SUBTYPES:
                friendly_minions.append(npc)
        else:
            if subtype == 21:
                enemy_tower = npc
            elif subtype == 24:
                enemy_crystal = npc
            elif subtype not in ORGAN_SUBTYPES:
                enemy_minions.append(npc)

    if memory is not None:
        if main_tower is not None:
            memory["main_tower_seen"] = True
            memory["main_tower_loc"] = _unit_xy(main_tower)
            memory["main_tower_hp"] = float(main_tower.get("hp", 0.0))
            memory["main_tower_max_hp"] = float(main_tower.get("max_hp", 0.0))
        elif saw_dead_main_tower or memory.get("main_tower_seen"):
            memory["main_tower_hp"] = 0.0

        if main_spring is not None:
            memory["main_spring_loc"] = _unit_xy(main_spring)

        if enemy_tower is not None:
            memory["enemy_tower_seen"] = True
            memory["enemy_tower_loc"] = _unit_xy(enemy_tower)
            memory["enemy_tower_hp"] = float(enemy_tower.get("hp", 0.0))
            memory["enemy_tower_max_hp"] = float(enemy_tower.get("max_hp", 0.0))
        else:
            known_et_loc = memory.get("enemy_tower_loc") or landmarks["enemy_tower"]
            if saw_dead_enemy_tower or (memory.get("enemy_tower_seen") and _dist(main_pos, known_et_loc) < 12000.0):
                memory["enemy_tower_hp"] = 0.0

        if enemy_crystal is not None:
            memory["enemy_crystal_loc"] = _unit_xy(enemy_crystal)

    # Resolve positions with memory fallback
    home_pos = (
        _unit_xy(main_spring)
        if main_spring is not None
        else (memory.get("main_spring_loc") if memory and "main_spring_loc" in memory else landmarks["home"])
    )
    main_tower_pos = (
        _unit_xy(main_tower)
        if main_tower is not None
        else (memory.get("main_tower_loc") if memory and "main_tower_loc" in memory else landmarks["main_tower"])
    )
    enemy_tower_pos = (
        _unit_xy(enemy_tower)
        if enemy_tower is not None
        else (memory.get("enemy_tower_loc") if memory and "enemy_tower_loc" in memory else landmarks["enemy_tower"])
    )
    enemy_crystal_pos = (
        _unit_xy(enemy_crystal)
        if enemy_crystal is not None
        else (memory.get("enemy_crystal_loc") if memory and "enemy_crystal_loc" in memory else landmarks["enemy_crystal"])
    )

    # Tower alive and HP ratios
    main_tower_hp = (
        float(main_tower.get("hp", 0.0))
        if main_tower is not None
        else (0.0 if saw_dead_main_tower else (memory.get("main_tower_hp", 12000.0) if memory else 12000.0))
    )
    main_tower_max = (
        float(main_tower.get("max_hp", 0.0))
        if main_tower is not None
        else (memory.get("main_tower_max_hp", 12000.0) if memory else 12000.0)
    )
    main_tower_hp_ratio = main_tower_hp / main_tower_max if main_tower_max > 0.0 else 1.0

    enemy_tower_hp = (
        float(enemy_tower.get("hp", 0.0))
        if enemy_tower is not None
        else (0.0 if saw_dead_enemy_tower else (memory.get("enemy_tower_hp", 12000.0) if memory else 12000.0))
    )

    # Check tower attack aggro on main hero
    tower_targeting_me = False
    if enemy_tower is not None and main_rtid is not None:
        if enemy_tower.get("attack_target") == main_rtid:
            tower_targeting_me = True

    # Active target organ for pushing
    enemy_tower_alive = enemy_tower_hp > 0.0
    target_organ_pos = enemy_tower_pos if enemy_tower_alive else enemy_crystal_pos
    dist_to_target_organ = _dist(main_pos, target_organ_pos)
    organ_edge_dist = max(0.0, dist_to_target_organ - ORGAN_RADIUS)

    nearest_minion_dist = min(
        (_dist(main_pos, _unit_xy(m)) for m in enemy_minions),
        default=float("inf"),
    )

    # Minions under enemy tower
    minions_under_tower = 0
    if enemy_tower_alive:
        minions_under_tower = sum(
            1 for m in friendly_minions if _dist(_unit_xy(m), enemy_tower_pos) < TOWER_DANGER_RANGE
        )

    tower_danger = tower_targeting_me or (
        enemy_tower_alive
        and _dist(main_pos, enemy_tower_pos) < TOWER_DANGER_RANGE
        and minions_under_tower == 0
    )

    # Cake search
    own_cake_available = False
    own_cake_loc = None
    dist_to_own_cake = float("inf")
    cakes = frame_state.get("cakes") or []
    for c in cakes:
        cloc = c.get("collider", {}).get("location") or c.get("location")
        if cloc is None:
            continue
        c_pos = (float(cloc.get("x", 0.0)), float(cloc.get("z", 0.0)))
        if _is_sentinel(c_pos):
            continue
        # Own cake is closer to our tower than enemy tower
        if _dist(c_pos, main_tower_pos) < _dist(c_pos, enemy_tower_pos):
            own_cake_available = True
            own_cake_loc = c_pos
            dist_to_own_cake = _dist(main_pos, c_pos)
            if memory is not None:
                memory["own_cake_loc"] = c_pos
            break

    # Enemy minions near our main tower
    enemy_minions_at_main_tower = sum(
        1 for m in enemy_minions if _dist(_unit_xy(m), main_tower_pos) < TOWER_DANGER_RANGE
    )

    return {
        "alive": main_hp > 0.0,
        "main_pos": main_pos,
        "main_hp_ratio": main_hp_ratio,
        "enemy_pos": enemy_pos,
        "enemy_visible": enemy_visible,
        "enemy_hp_ratio": enemy_hp_ratio,
        "enemy_hero_dist": enemy_hero_dist,
        "nearest_target_dist": min(enemy_hero_dist, nearest_minion_dist, organ_edge_dist),
        "nearest_minion_dist": nearest_minion_dist,
        "tower_danger": tower_danger,
        "tower_targeting_me": tower_targeting_me,
        "minions_under_tower": minions_under_tower,
        "home_pos": home_pos,
        "dist_to_home": _dist(main_pos, home_pos),
        "main_tower_pos": main_tower_pos,
        "main_tower_hp_ratio": main_tower_hp_ratio,
        "enemy_minions_at_main_tower": enemy_minions_at_main_tower,
        "target_organ_pos": target_organ_pos,
        "dist_to_target_organ": dist_to_target_organ,
        "enemy_tower_alive": enemy_tower_alive,
        "own_cake_available": own_cake_available,
        "own_cake_loc": own_cake_loc,
        "dist_to_own_cake": dist_to_own_cake,
        "main_camp": main_camp,
    }


def _can_execute_enemy(hp: float, ehp: float, enemy_d: float) -> bool:
    return (
        enemy_d < RETREAT_HERO_DIST
        and ehp < EXECUTE_HP
        and hp >= MIN_EXECUTE_HP
        and hp >= ehp - 0.10
    )


def _main_tower_urgent(state: Dict[str, Any]) -> bool:
    hp_ratio = state["main_tower_hp_ratio"]
    minions_near = state["enemy_minions_at_main_tower"] > 0
    return hp_ratio < TOWER_URGENT_HP_RATIO or (minions_near and hp_ratio < 0.60)


def _needs_combat_retreat(state: Dict[str, Any]) -> bool:
    if not state["alive"]:
        return False
    if state["tower_danger"]:
        return True
    hp = state["main_hp_ratio"]
    ehp = state["enemy_hp_ratio"]
    enemy_d = state["enemy_hero_dist"]
    if _can_execute_enemy(hp, ehp, enemy_d):
        return False
    close_duel = hp >= EXECUTE_HP and ehp < hp + DISENGAGE_HP_GAP
    if close_duel:
        return False
    critical = hp < EXECUTE_HP and enemy_d < RETREAT_HERO_DIST
    low_outmatched = (
        hp < DISENGAGE_HP and enemy_d < RETREAT_HERO_DIST and ehp >= hp + DISENGAGE_HP_GAP
    )
    return critical or low_outmatched


def _recovery_mode(state: Dict[str, Any]) -> Optional[str]:
    if not state["alive"]:
        return None
    if state["tower_danger"]:
        return "fountain"
    hp = state["main_hp_ratio"]
    ehp = state["enemy_hp_ratio"]
    enemy_d = state["enemy_hero_dist"]
    if _can_execute_enemy(hp, ehp, enemy_d):
        return None

    tower_urgent = _main_tower_urgent(state)
    done_hp = HOME_URGENT_DONE_HP if tower_urgent else HOME_DONE_HP
    if state["dist_to_home"] <= HOME_BASE_DIST and hp < done_hp:
        return "fountain"

    own_cake = state["own_cake_available"]
    d_cake = state["dist_to_own_cake"]
    combat_retreat = _needs_combat_retreat(state)

    if hp < HOME_HP or combat_retreat:
        if own_cake and hp >= CRITICAL_HOME_HP and d_cake <= 16000.0 and enemy_d > 6500.0:
            return "cake"
        if tower_urgent and hp >= CRITICAL_HOME_HP and not combat_retreat:
            return None
        return "fountain"

    if own_cake and hp < CAKE_HP and d_cake <= CAKE_MAX_DETOUR and enemy_d > 8500.0:
        return "cake"

    return None


def _latch_home(state: Dict[str, Any], rec_mode: Optional[str], memory: Dict[str, Any]) -> Optional[str]:
    """Commit to a low-HP fountain trip until healed.

    A trip that starts below HOME_HP keeps "fountain" (a cake detour is still allowed)
    until HP reaches HOME_DONE_HP, or HOME_URGENT_DONE_HP while our tower is urgent;
    without the latch the corridor turned the hero around halfway home. Tower-danger
    escapes at higher HP do not start a trip.
    """
    hp = state["main_hp_ratio"]
    if rec_mode == "fountain" and hp < HOME_HP:
        memory["returning_home"] = True
    if not memory.get("returning_home"):
        return rec_mode
    done_hp = HOME_URGENT_DONE_HP if _main_tower_urgent(state) else HOME_DONE_HP
    if hp >= done_hp:
        memory["returning_home"] = False
        return rec_mode
    return "cake" if rec_mode == "cake" else "fountain"


def _under_hero_pressure(state: Dict[str, Any], memory: Optional[Dict[str, Any]]) -> bool:
    """Visible enemy hero within HERO_PRESSURE_DIST and our HP fell over the last steps (not tower fire)."""
    if memory is None:
        return False
    history = memory.setdefault("hp_history", [])
    history.append(state["main_hp_ratio"])
    del history[: -(HERO_PRESSURE_STEPS + 1)]
    return (
        state["enemy_visible"]
        and state["enemy_hero_dist"] <= HERO_PRESSURE_DIST
        and not state["tower_danger"]
        and len(history) == HERO_PRESSURE_STEPS + 1
        and history[-1] < history[0]
    )


def _lock_move_bins(refined: np.ndarray, raw: np.ndarray, ux: float, uz: float) -> bool:
    """Lock Move X [12:28] and Move Z [28:44] to the single discrete bin along (ux, uz)."""
    if ux == 0.0 and uz == 0.0:
        bx, bz = 8, 8
    else:
        bx = int(np.clip(round(8.0 + 7.0 * ux), 1, 15))
        bz = int(np.clip(round(8.0 + 7.0 * uz), 1, 15))

    if raw[12 + bx] <= 0.0 or raw[28 + bz] <= 0.0:
        return False

    refined[12:28] = 0.0
    refined[12 + bx] = raw[12 + bx]
    refined[28:44] = 0.0
    refined[28 + bz] = raw[28 + bz]
    return True


def _lock_skill_bins(refined: np.ndarray, raw: np.ndarray, ux: float, uz: float) -> bool:
    """Lock Skill X [44:60] and Skill Z [60:76] to the single discrete bin along (ux, uz)."""
    if ux == 0.0 and uz == 0.0:
        bx, bz = 8, 8
    else:
        bx = int(np.clip(round(8.0 + 7.0 * ux), 1, 15))
        bz = int(np.clip(round(8.0 + 7.0 * uz), 1, 15))

    if raw[44 + bx] <= 0.0 or raw[60 + bz] <= 0.0:
        return False

    refined[44:60] = 0.0
    refined[44 + bx] = raw[44 + bx]
    refined[60:76] = 0.0
    refined[60 + bz] = raw[60 + bz]
    return True


def _world_delta_to_action_unit(
    src_pos: Tuple[float, float],
    dst_pos: Tuple[float, float],
    main_camp: Any,
) -> Optional[Tuple[float, float]]:
    """Unit direction in action space (GameCore mirrors X/Z actions for Camp 2)."""
    dx = dst_pos[0] - src_pos[0]
    dz = dst_pos[1] - src_pos[1]
    norm = math.hypot(dx, dz)
    if norm <= 1e-6:
        return None
    sign = -1.0 if main_camp in (2, "PLAYERCAMP_2") else 1.0
    return (sign * dx / norm, sign * dz / norm)


def _guide_flash(
    refined: np.ndarray,
    raw: np.ndarray,
    state: Dict[str, Any],
) -> None:
    """Flash (Button 8) tactical guidance: escape vs execute vs baseline block."""
    raw_btn = raw[: LABEL_SIZES[0]]
    if raw_btn[BUTTON_CHOSEN] <= 0.0:
        # Flash is on cooldown or disabled by engine
        return

    main_pos = state["main_pos"]
    home_pos = state["home_pos"]
    main_camp = state.get("main_camp", 1)
    hp = state["main_hp_ratio"]
    ehp = state["enemy_hp_ratio"]
    enemy_d = state["enemy_hero_dist"]
    tower_danger = state["tower_danger"]
    tower_targeting_me = state["tower_targeting_me"]

    target_offset = sum(LABEL_SIZES[:-1])
    btn8_target_offset = target_offset + BUTTON_CHOSEN * LABEL_SIZES[-1]

    # Helper to lock Flash target to Self (2) or None (0)
    def _lock_flash_target():
        if raw[btn8_target_offset + TARGET_SELF] > 0.0:
            refined[btn8_target_offset : btn8_target_offset + 9] = 0.0
            refined[btn8_target_offset + TARGET_SELF] = raw[btn8_target_offset + TARGET_SELF]
        elif raw[btn8_target_offset + TARGET_NONE] > 0.0:
            refined[btn8_target_offset : btn8_target_offset + 9] = 0.0
            refined[btn8_target_offset + TARGET_NONE] = raw[btn8_target_offset + TARGET_NONE]

    # 1. Escape Flash: endangered by tower or critical close duel
    escape_needed = (
        (tower_targeting_me and hp < 0.65)
        or (tower_danger and hp < 0.55)
        or (hp < EXECUTE_HP and enemy_d < FLASH_ESCAPE_HERO_DIST and not _can_execute_enemy(hp, ehp, enemy_d))
    )

    if escape_needed:
        unit = _world_delta_to_action_unit(main_pos, home_pos, main_camp)
        if unit is not None and _lock_skill_bins(refined, raw, unit[0], unit[1]):
            _lock_flash_target()
            refined[BUTTON_CHOSEN] = raw[BUTTON_CHOSEN]
            return

    # 2. Execute Flash: low-HP enemy within flash-engage window while healthy
    execute_window = (
        state["enemy_visible"]
        and ehp < FLASH_EXECUTE_ENEMY_HP
        and hp >= 0.40
        and not tower_danger
        and FLASH_EXECUTE_MIN_DIST < enemy_d <= FLASH_EXECUTE_MAX_DIST
        and state["enemy_pos"] is not None
    )

    if execute_window:
        unit = _world_delta_to_action_unit(main_pos, state["enemy_pos"], main_camp)
        if unit is not None and _lock_skill_bins(refined, raw, unit[0], unit[1]):
            _lock_flash_target()
            refined[BUTTON_CHOSEN] = raw[BUTTON_CHOSEN]
            return

    # 3. Otherwise: zero Button 8 to suppress accidental flash-burn
    refined[BUTTON_CHOSEN] = 0.0


def apply_tactical_mask(
    legal_action: Any,
    frame_state: Optional[Dict[str, Any]],
    main_camp: int,
    memory: Optional[Dict[str, Any]] = None,
    enable_movement: bool = True,
    enable_flash_guidance: bool = True,
    trace: Optional[Dict[str, Any]] = None,
) -> Any:
    """Return a refined copy of the 184-dim V63 legal_action vector.

    When ``trace`` is a dict it is filled with the branch taken this step
    (``dead``, ``retreat``, ``rec`` = None/"cake"/"fountain", ``corridor``,
    ``tower_danger``, ``hero_pressure``, ``recall_legal``) so replays can explain
    why the policy moved as it did.
    """
    raw = np.asarray(legal_action, dtype=np.float32)
    if raw.shape != (EXPECTED_LEGAL_LEN,):
        return legal_action

    state = _extract_tactical_state(frame_state, main_camp, memory)
    if state is None:
        return raw.copy()

    refined = raw.copy()
    btn = refined[: LABEL_SIZES[0]]
    raw_btn = raw[: LABEL_SIZES[0]]

    target_offset = sum(LABEL_SIZES[:-1])
    target_mat = refined[target_offset:].reshape(LABEL_SIZES[0], LABEL_SIZES[-1])
    raw_target_mat = raw[target_offset:].reshape(LABEL_SIZES[0], LABEL_SIZES[-1])

    if trace is not None:
        trace.update(dead=not state["alive"], retreat=False, rec=None, corridor=False,
                     tower_danger=bool(state["tower_danger"]), recall_legal=bool(raw_btn[BUTTON_RECALL] > 0))

    # Dead state: zero all active combat buttons; a death ends any fountain trip
    if not state["alive"]:
        if memory is not None:
            memory["returning_home"] = False
            memory["hp_history"] = []
        for b in (BUTTON_ATTACK, BUTTON_SKILL1, BUTTON_SKILL2, BUTTON_SKILL3, BUTTON_HEAL, BUTTON_CHOSEN):
            btn[b] = 0.0
        if btn.sum() <= 0.0 and raw_btn.sum() > 0.0:
            btn[:] = raw_btn
        return refined

    hero_pressure = _under_hero_pressure(state, memory)
    if trace is not None:
        trace["hero_pressure"] = hero_pressure

    combat_legal = any(btn[b] > 0.0 for b in COMBAT_BUTTONS)
    if combat_legal:
        # Zero idle/useless buttons
        for idle_b in (BUTTON_NONE_0, BUTTON_NONE_1, BUTTON_RECALL):
            btn[idle_b] = 0.0

        if state["main_hp_ratio"] >= HEAL_HP_THRESHOLD:
            btn[BUTTON_HEAL] = 0.0

        # Target range gating
        nearest = state["nearest_target_dist"]
        if nearest > ATTACK_RADIUS:
            btn[BUTTON_ATTACK] = 0.0
        if nearest > SKILL_RADIUS:
            btn[BUTTON_SKILL1] = 0.0
            btn[BUTTON_SKILL3] = 0.0
        if nearest > SKILL2_RADIUS:
            btn[BUTTON_SKILL2] = 0.0

        # Combat retreat
        in_combat_retreat = _needs_combat_retreat(state)
        if trace is not None:
            trace["retreat"] = in_combat_retreat
        if in_combat_retreat:
            # Under severe tower danger or panic retreat: disengage offensive casts
            for off_b in OFFENSE_BUTTONS:
                btn[off_b] = 0.0
                target_mat[off_b, TARGET_ENEMY_HERO] = 0.0
                target_mat[off_b, TARGET_TOWER] = 0.0
                if target_mat[off_b].sum() <= 0.0 and raw_target_mat[off_b].sum() > 0.0:
                    target_mat[off_b] = raw_target_mat[off_b]

        # Hero pressure: trade with the hero or step away, never farm minions/tower under fire
        if hero_pressure:
            if state["enemy_hero_dist"] <= ATTACK_RADIUS:
                for off_b in OFFENSE_BUTTONS:
                    target_mat[off_b, NON_HERO_TARGETS] = 0.0
                    if target_mat[off_b].sum() <= 0.0:
                        btn[off_b] = 0.0
                        target_mat[off_b] = raw_target_mat[off_b]
            else:
                btn[BUTTON_ATTACK] = 0.0

        # Movement locking: recovery vs outbound corridor
        if enable_movement:
            rec_mode = _recovery_mode(state)
            if memory is not None:
                rec_mode = _latch_home(state, rec_mode, memory)
            if trace is not None:
                trace["rec"] = rec_mode
            if rec_mode == "cake" and state["own_cake_loc"] is not None:
                unit = _world_delta_to_action_unit(state["main_pos"], state["own_cake_loc"], main_camp)
                if unit is not None:
                    _lock_move_bins(refined, raw, unit[0], unit[1])
            elif rec_mode == "fountain":
                if state["dist_to_home"] <= FOUNTAIN_STAND_DIST:
                    _lock_move_bins(refined, raw, 0.0, 0.0)
                else:
                    unit = _world_delta_to_action_unit(state["main_pos"], state["home_pos"], main_camp)
                    if unit is not None:
                        _lock_move_bins(refined, raw, unit[0], unit[1])
            elif (
                rec_mode is None
                and not in_combat_retreat
                and state["enemy_hero_dist"] > CORRIDOR_HERO_DIST
                and state["nearest_minion_dist"] > CORRIDOR_MINION_DIST
                and state["main_hp_ratio"] >= CORRIDOR_MIN_HP
            ):
                # Outbound push corridor toward enemy tower / crystal
                crystal_phase = not state["enemy_tower_alive"]
                wait_dist = 10800.0 if crystal_phase else CORRIDOR_WAIT_TOWER_DIST
                push_dist = 8500.0 if crystal_phase else CORRIDOR_PUSH_TOWER_DIST
                stop_dist = push_dist if state["minions_under_tower"] > 0 else wait_dist
                if state["dist_to_target_organ"] > stop_dist:
                    unit = _world_delta_to_action_unit(
                        state["main_pos"], state["target_organ_pos"], main_camp
                    )
                    if unit is not None:
                        locked = _lock_move_bins(refined, raw, unit[0], unit[1])
                        if trace is not None:
                            trace["corridor"] = locked

        # Flash tactical gating and directional guidance
        if enable_flash_guidance:
            _guide_flash(refined, raw, state)

    # Fallback safety: never leave button head completely empty
    if btn.sum() <= 0.0 and raw_btn.sum() > 0.0:
        if raw_btn[BUTTON_MOVE] > 0.0:
            btn[BUTTON_MOVE] = raw_btn[BUTTON_MOVE]
        else:
            btn[:] = raw_btn

    return refined
