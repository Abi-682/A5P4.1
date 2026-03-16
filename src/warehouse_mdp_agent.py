"""Problem 4.1 (Parts 1-2): MDP Agent core for the 3x4 grid.

Grid setup:
- Coordinates are (x, y), with x in [1..4], y in [1..3]
- Wall: (2, 2)
- Goal terminal: (4, 3) with reward +1
- Hazard terminal: (4, 2) with reward -1
- Living reward: -0.04

Dynamics:
- Intended action succeeds with probability 0.8
- Each perpendicular drift occurs with probability 0.1
- Invalid moves (off-grid or into wall) leave the state unchanged
"""

import random
from collections import Counter

# Grid dimensions
WIDTH, HEIGHT = 4, 3

# Terminal states and rewards
GOAL = (4, 3)
HAZARD = (4, 2)
TERMINALS = {GOAL: 1.0, HAZARD: -1.0}

# Living reward for non-terminal states
LIVING_REWARD = -0.04

# Wall position
WALL = (2, 2)

# All valid states (all grid cells except the wall)
STATES = [
    (x, y)
    for x in range(1, WIDTH + 1)
    for y in range(1, HEIGHT + 1)
    if (x, y) != WALL
]

# Action set and displacement vectors
ACTIONS = {
    "North": (0, 1),
    "South": (0, -1),
    "East": (1, 0),
    "West": (-1, 0),
}

ARROWS = {"North": "^", "South": "v", "East": ">", "West": "<"}


def reward(state):
    """R(s): immediate reward for being in state s."""
    if state in TERMINALS:
        return TERMINALS[state]
    return LIVING_REWARD


def get_perpendicular(action):
    """Return the two actions perpendicular to the given action."""
    if action in ("North", "South"):
        return ["West", "East"]
    return ["North", "South"]


def attempt_move(state, action):
    """Attempt one deterministic move.

    If the move leaves the grid or hits the wall, return the original state.
    """
    dx, dy = ACTIONS[action]
    nx, ny = state[0] + dx, state[1] + dy
    if 1 <= nx <= WIDTH and 1 <= ny <= HEIGHT and (nx, ny) != WALL:
        return (nx, ny)
    return state


def transitions(state, action):
    """T(s' | s, a): return a dict {next_state: probability}."""
    if state in TERMINALS:
        return {}

    outcomes = {}

    intended = attempt_move(state, action)
    outcomes[intended] = outcomes.get(intended, 0.0) + 0.8

    for perp in get_perpendicular(action):
        drifted = attempt_move(state, perp)
        outcomes[drifted] = outcomes.get(drifted, 0.0) + 0.1

    return outcomes


def value_iteration(gamma=0.99, epsilon=1e-6):
    """Run value iteration.

    Returns:
        (V, iterations)
    where V is a converged state-value dict and iterations is the
    number of sweeps required to converge.
    """
    V = {s: 0.0 for s in STATES}
    iteration = 0

    while True:
        V_new = {}
        delta = 0.0

        for s in STATES:
            if s in TERMINALS:
                V_new[s] = reward(s)
                continue

            best_value = float("-inf")
            for a in ACTIONS:
                expected = sum(
                    prob * V[s_next] for s_next, prob in transitions(s, a).items()
                )
                best_value = max(best_value, expected)

            V_new[s] = reward(s) + gamma * best_value
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        iteration += 1

        if delta < epsilon:
            break

    return V, iteration


def extract_policy(V, gamma=0.99):
    """Extract a greedy policy from a converged value function."""
    policy = {}

    for s in STATES:
        if s in TERMINALS:
            policy[s] = None
            continue

        best_action = None
        best_value = float("-inf")

        for a in ACTIONS:
            expected = sum(
                prob * V[s_next] for s_next, prob in transitions(s, a).items()
            )
            q_sa = reward(s) + gamma * expected
            if q_sa > best_value:
                best_value = q_sa
                best_action = a

        policy[s] = best_action

    return policy


def simulate_step(state, action):
    """Part 1: sample one next state from T(s' | s, a)."""
    dist = transitions(state, action)
    states = list(dist.keys())
    probs = list(dist.values())
    return random.choices(states, weights=probs, k=1)[0]


def run_episode(policy, start=(1, 1), max_steps=100):
    """Part 2: simulate one full episode under a policy.

    Returns:
        trajectory, total_reward, outcome
    outcome is one of: "goal", "hazard", "timeout".
    """
    state = start
    trajectory = [state]
    total_reward = reward(state)

    for _ in range(max_steps):
        if state in TERMINALS:
            break

        action = policy[state]
        state = simulate_step(state, action)
        trajectory.append(state)
        total_reward += reward(state)

    if state == GOAL:
        outcome = "goal"
    elif state == HAZARD:
        outcome = "hazard"
    else:
        outcome = "timeout"

    return trajectory, total_reward, outcome


def _verify_simulate_step():
    """Print empirical frequencies for simulate_step((3,1), 'North')."""
    random.seed(42)
    counts = Counter()
    for _ in range(10_000):
        counts[simulate_step((3, 1), "North")] += 1

    print("Empirical transition frequencies from (3,1), action North:")
    for s, c in sorted(counts.items()):
        print(f"  {s}: {c / 10_000:.3f}")


def _run_optimal_policy_demo():
    """Run 1,000 episodes with the optimal policy and print summary stats."""
    random.seed(42)

    V, num_iters = value_iteration(gamma=0.99)
    optimal_policy = extract_policy(V, gamma=0.99)

    print(f"\nValue iteration converged in {num_iters} iterations.\n")

    results = [run_episode(optimal_policy) for _ in range(1000)]
    outcomes = [r[2] for r in results]
    rewards = [r[1] for r in results]

    goal_rate = outcomes.count("goal") / 1000
    hazard_rate = outcomes.count("hazard") / 1000
    avg_reward = sum(rewards) / 1000

    print(f"Goal reached:  {goal_rate:.3f}")
    print(f"Hazard hit:    {hazard_rate:.3f}")
    print(f"Avg reward:    {avg_reward:.3f}")


if __name__ == "__main__":
    _verify_simulate_step()
    _run_optimal_policy_demo()
