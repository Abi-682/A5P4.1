"""
Warehouse MDP Agent  (Assignment – Tasks 1-5)
==============================================
MDP framework (Section 4.4: Value Iteration for the Warehouse),
extended with an environment simulator, agent loop, policy comparison,
discount-factor sweep, and a harder warehouse.

Grid layout  (4 rows × 3 cols, 1-indexed, North = toward row 1)
─────────────────────────────────────────────
  Col:   1    2    3
Row 1:   .    .    .
Row 2:   .    .    .
Row 3:   .    .    H   ← Hazard  (reward −1, terminal)
Row 4:   .    .    G   ← Goal    (reward +1, terminal)

Start: (1, 1)  |  Living reward: −0.04  |  Default γ: 0.9
Stochastic motion: 80 % intended, 10 % each perpendicular (walls → bounce).
"""

import random
from collections import defaultdict

# ────────────────────────────────────────────────────────────────────────────
# MDP constants
# ────────────────────────────────────────────────────────────────────────────

ROWS, COLS    = 4, 3
GOAL          = (4, 3)   # terminal, reward +1
HAZARD        = (3, 3)   # terminal, reward −1
LIVING_R      = -0.04
DEFAULT_GAMMA = 0.9

ACTIONS = ["North", "South", "East", "West"]

_DELTA = {"North": (-1, 0), "South": (1, 0), "East": (0, 1), "West": (0, -1)}
_PERP  = {
    "North": ["West",  "East"],
    "South": ["West",  "East"],
    "East" : ["North", "South"],
    "West" : ["North", "South"],
}

# ────────────────────────────────────────────────────────────────────────────
# MDP core  (Section 4.4)
# ────────────────────────────────────────────────────────────────────────────

def all_states(hazards=None):
    """Return every non-terminal grid state."""
    hazards = hazards if hazards is not None else {HAZARD}
    return [
        (r, c)
        for r in range(1, ROWS + 1)
        for c in range(1, COLS + 1)
        if (r, c) != GOAL and (r, c) not in hazards
    ]


def _move(state, action):
    """Deterministic step; hitting a wall returns the original state."""
    r, c   = state
    dr, dc = _DELTA[action]
    nr, nc = r + dr, c + dc
    return (nr, nc) if 1 <= nr <= ROWS and 1 <= nc <= COLS else state


def transitions(state, action):
    """
    Return list of (probability, next_state) tuples.
    80 % intended direction, 10 % each perpendicular direction.
    Outcomes that map to the same cell are merged.
    """
    raw = {_move(state, action): 0.8}
    for perp in _PERP[action]:
        ns = _move(state, perp)
        raw[ns] = raw.get(ns, 0.0) + 0.1
    return [(prob, ns) for ns, prob in raw.items()]


def reward(state, action, next_state, hazard_rewards=None):
    """R(s, a, s') — reward received when transitioning to next_state."""
    if next_state == GOAL:
        return 1.0
    if hazard_rewards and next_state in hazard_rewards:
        return hazard_rewards[next_state]
    return LIVING_R


def is_terminal(state, hazards=None):
    hazards = hazards if hazards is not None else {HAZARD}
    return state == GOAL or state in hazards


def value_iteration(gamma=DEFAULT_GAMMA, theta=1e-8, hazards=None):
    """
    Run value iteration and return the converged value function V.

    Parameters
    ----------
    gamma   : discount factor
    theta   : convergence threshold (max change per sweep)
    hazards : set of hazard states (default: {HAZARD})
    """
    hazards = hazards if hazards is not None else {HAZARD}
    hrew    = {h: -1.0 for h in hazards}
    S       = all_states(hazards)
    V       = {s: 0.0 for s in S}

    while True:
        delta = 0.0
        for s in S:
            q_max = max(
                sum(p * (reward(s, a, ns, hrew) + gamma * V.get(ns, 0.0))
                    for p, ns in transitions(s, a))
                for a in ACTIONS
            )
            delta = max(delta, abs(V[s] - q_max))
            V[s]  = q_max
        if delta < theta:
            break
    return V


def extract_policy(V, hazards=None):
    """Return the greedy policy π*(s) = argmax_a Q(s, a) from value function V."""
    hazards = hazards if hazards is not None else {HAZARD}
    hrew    = {h: -1.0 for h in hazards}
    return {
        s: max(
            ACTIONS,
            key=lambda a: sum(
                p * (reward(s, a, ns, hrew) + V.get(ns, 0.0))
                for p, ns in transitions(s, a)
            ),
        )
        for s in all_states(hazards)
    }


# ────────────────────────────────────────────────────────────────────────────
# Task 1 – Environment simulator
# ────────────────────────────────────────────────────────────────────────────

def simulate_step(state, action):
    """
    Sample the next state s' ~ T(· | state, action).

    Uses random.choices to draw one outcome from the transition distribution
    returned by transitions(state, action).
    """
    trans       = transitions(state, action)
    probs       = [p  for p, _  in trans]
    next_states = [ns for _, ns in trans]
    return random.choices(next_states, weights=probs, k=1)[0]


def _verify_simulate_step(n=10_000):
    """
    Verify simulate_step by calling simulate_step((3, 1), 'North') n times
    and checking that empirical frequencies approximate 80 / 10 / 10.
    """
    print("=== Task 1: Verifying simulate_step((3, 1), 'North') ===")
    counts = defaultdict(int)
    for _ in range(n):
        counts[simulate_step((3, 1), "North")] += 1
    for ns in sorted(counts):
        pct      = counts[ns] / n * 100
        expected = "~80 %" if ns == (2, 1) else "~10 %"
        print(f"  → {ns}: {pct:5.1f}%  (expected {expected})")
    print()


# ────────────────────────────────────────────────────────────────────────────
# Task 2 – Agent loop
# ────────────────────────────────────────────────────────────────────────────

def run_episode(policy, start=(1, 1), max_steps=100, hazards=None):
    """
    Simulate one full episode under *policy*.

    At each step the agent looks up π*(s) in the policy, calls simulate_step,
    accumulates the undiscounted reward, and records the trajectory.
    Stops when a terminal state is reached or max_steps is exceeded.

    Returns
    -------
    trajectory   : list of states visited (including start and final state)
    total_reward : undiscounted sum of rewards collected
    outcome      : 'goal' | 'hazard' | 'timeout'
    """
    hazards = hazards if hazards is not None else {HAZARD}
    hrew    = {h: -1.0 for h in hazards}

    state        = start
    trajectory   = [state]
    total_reward = 0.0

    for _ in range(max_steps):
        if is_terminal(state, hazards):
            break
        action     = policy[state]
        next_state = simulate_step(state, action)
        total_reward += reward(state, action, next_state, hrew)
        state = next_state
        trajectory.append(state)

    if state == GOAL:
        outcome = "goal"
    elif state in hazards:
        outcome = "hazard"
    else:
        outcome = "timeout"

    return trajectory, total_reward, outcome


def _run_n_episodes(policy, n=1000, hazards=None):
    """Run n episodes and aggregate statistics."""
    counts  = {"goal": 0, "hazard": 0, "timeout": 0}
    total_r = 0.0
    for _ in range(n):
        _, r, outcome = run_episode(policy, hazards=hazards)
        counts[outcome] += 1
        total_r         += r
    return {
        "goal_rate":   counts["goal"]    / n,
        "hazard_rate": counts["hazard"]  / n,
        "timeout_rate":counts["timeout"] / n,
        "avg_reward":  total_r           / n,
    }


def _print_stats(label, stats):
    print(f"  {label}")
    print(f"    Goal reached  : {stats['goal_rate']   * 100:6.1f} %")
    print(f"    Hazard hit    : {stats['hazard_rate'] * 100:6.1f} %")
    print(f"    Timed out     : {stats['timeout_rate']* 100:6.1f} %")
    print(f"    Avg reward    : {stats['avg_reward']:8.3f}")
    print()


# ────────────────────────────────────────────────────────────────────────────
# Task 3 – Naive greedy policy
# ────────────────────────────────────────────────────────────────────────────

def _naive_action(state, goal=GOAL):
    """
    Always move one step closer to *goal*, ignoring hazards.
    Priority: rows first (North/South), then columns (East/West).
    """
    r, c   = state
    gr, gc = goal
    if gr > r: return "South"
    if gr < r: return "North"
    if gc > c: return "East"
    return "West"


def make_naive_policy(hazards=None):
    """Build a naive greedy policy dict for all non-terminal states."""
    hazards = hazards if hazards is not None else {HAZARD}
    return {s: _naive_action(s) for s in all_states(hazards)}


# ────────────────────────────────────────────────────────────────────────────
# Display helper
# ────────────────────────────────────────────────────────────────────────────

_SYM = {"North": "↑", "South": "↓", "East": "→", "West": "←"}


def print_policy(policy, hazards=None, title="Policy"):
    """Print the policy as an ASCII grid (row 1 = top)."""
    hazards = hazards if hazards is not None else {HAZARD}
    bar     = "  " + "─" * (COLS * 5 + 1)
    print(f"  {title}")
    print(bar)
    for r in range(1, ROWS + 1):
        row = "  │"
        for c in range(1, COLS + 1):
            s = (r, c)
            if s == GOAL:
                cell = " G  "
            elif s in hazards:
                cell = " H  "
            else:
                cell = f" {_SYM.get(policy.get(s, '?'), '?')}  "
            row += cell + "│"
        print(row)
    print(bar)
    print()


# ────────────────────────────────────────────────────────────────────────────
# Main – execute all five tasks
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    # ── Task 1 ──────────────────────────────────────────────────────────────
    _verify_simulate_step()

    # ── Task 2 ──────────────────────────────────────────────────────────────
    print("=== Task 2: Optimal Policy (γ = 0.9) — 1 000 episodes ===")
    V_opt      = value_iteration(gamma=0.9)
    policy_opt = extract_policy(V_opt)
    print_policy(policy_opt, title="Optimal policy  (γ = 0.9)")
    stats_opt = _run_n_episodes(policy_opt, n=1000)
    _print_stats("Results (optimal policy):", stats_opt)

    # ── Task 3 ──────────────────────────────────────────────────────────────
    print("=== Task 3: Naive Greedy Policy — 1 000 episodes ===")
    policy_naive = make_naive_policy()
    print_policy(policy_naive, title="Naive greedy policy")
    stats_naive = _run_n_episodes(policy_naive, n=1000)
    _print_stats("Results (naive policy):", stats_naive)

    print("  Head-to-head comparison:")
    print(f"    Goal rate   — Optimal: {stats_opt['goal_rate']   * 100:.1f} %"
          f"   Naive: {stats_naive['goal_rate']   * 100:.1f} %")
    print(f"    Hazard rate — Optimal: {stats_opt['hazard_rate'] * 100:.1f} %"
          f"   Naive: {stats_naive['hazard_rate'] * 100:.1f} %")
    print(f"    Avg reward  — Optimal: {stats_opt['avg_reward']:.3f}"
          f"   Naive: {stats_naive['avg_reward']:.3f}")
    print()

    # ── Task 4 ──────────────────────────────────────────────────────────────
    print("=== Task 4: Discount Factor Experiment (γ ∈ {0.1, 0.5, 0.9, 0.99}) ===")
    print(f"  {'γ':>6}  │  {'Goal %':>7}  │  {'Hazard %':>8}  │  {'Avg Reward':>10}")
    print("  " + "─" * 46)
    for gamma in (0.1, 0.5, 0.9, 0.99):
        V_g   = value_iteration(gamma=gamma)
        pol_g = extract_policy(V_g)
        st    = _run_n_episodes(pol_g, n=1000)
        print(
            f"  {gamma:>6.2f}  │ {st['goal_rate']*100:>7.1f}  │"
            f" {st['hazard_rate']*100:>8.1f}  │  {st['avg_reward']:>10.3f}"
        )
    print()
    print("  Observation: At low γ the agent is myopic and may rush toward")
    print("  the goal recklessly; as γ rises the agent becomes cautious and")
    print("  first consistently avoids the hazard at intermediate γ values.")
    print()

    # ── Task 5 ──────────────────────────────────────────────────────────────
    print("=== Task 5: Harder Warehouse — second hazard at (2, 3) ===")
    HAZARD2  = (2, 3)
    hazards2 = {HAZARD, HAZARD2}
    V_hard   = value_iteration(hazards=hazards2)
    policy_h = extract_policy(V_hard, hazards=hazards2)
    print_policy(policy_h, hazards=hazards2, title="Harder-warehouse policy")
    stats_hard = _run_n_episodes(policy_h, n=1000, hazards=hazards2)
    _print_stats("Results (harder warehouse):", stats_hard)

    print("  Comparison — original vs harder:")
    print(f"    Goal rate  — Original: {stats_opt['goal_rate']  * 100:.1f} %"
          f"   Harder: {stats_hard['goal_rate']  * 100:.1f} %")
    print(f"    Avg reward — Original: {stats_opt['avg_reward']:.3f}"
          f"   Harder: {stats_hard['avg_reward']:.3f}")
    print()
    print("  Analysis:")
    print("    Adding a second hazard at (2,3) blocks the right column from")
    print("    rows 2 and 3, forcing the agent to approach the goal along the")
    print("    bottom rows only.  The policy shifts to hug the left/bottom")
    print("    corridor, and the goal-reach rate may decrease slightly because")
    print("    stochastic slips now have fewer safe recovery paths.")
