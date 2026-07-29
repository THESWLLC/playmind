from playmind.learning import owned_state_key, reward_owned, OnlinePolicy, OWNED_ACTIONS


def test_owned_state_key_and_reward() -> None:
    a = {
        "player": {"x": 0, "y": 0, "hp": 0.8},
        "vision_player_hp": 0.8,
        "has_target": False,
        "in_combat": False,
        "motion": 1.0,
    }
    b = {
        "player": {"x": 0, "y": 0, "hp": 0.8},
        "vision_player_hp": 0.8,
        "has_target": True,
        "in_combat": True,
        "motion": 6.0,
    }
    assert "notgt" in owned_state_key(a)
    assert "tgt" in owned_state_key(b)
    r_tab = reward_owned(a, "target_nearest", b)
    r_atk = reward_owned(b, "attack", b)
    r_miss = reward_owned(a, "attack", a)
    assert r_tab > 0
    assert r_atk > r_miss


def test_owned_policy_updates() -> None:
    policy = OnlinePolicy(epsilon=0.0, key_fn=owned_state_key)
    obs = {
        "player": {"hp": 0.9},
        "vision_player_hp": 0.9,
        "has_target": True,
        "in_combat": True,
        "motion": 0,
    }
    nxt = dict(obs)
    policy.update(obs, "attack", 0.5, nxt, False, list(OWNED_ACTIONS))
    assert policy.q[owned_state_key(obs)]["attack"] > 0
    assert policy.choose(obs, list(OWNED_ACTIONS)) == "attack"
