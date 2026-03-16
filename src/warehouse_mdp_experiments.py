"""Problem 4.1 (Parts 3-5): Experiments for the MDP agent.

Runs:
- Part 3: optimal vs naive greedy policy
- Part 4: discount factor experiment
- Part 5: harder warehouse (second hazard at (2,3))
"""

import random

import warehouse_mdp_agent as mdp


def greedy_policy_action(state):
    """Move toward the goal by Manhattan direction, ignoring hazards."""
    gx, gy = mdp.GOAL
    sx, sy = state

    if sx < gx:
        return "East"
    if sx > gx:
        return "West"
    if sy < gy:
        return "North"
    return "South"


def build_greedy_policy():
    policy = {s: greedy_policy_action(s) for s in mdp.STATES if s not in mdp.TERMINALS}
    policy[mdp.GOAL] = None
    policy[mdp.HAZARD] = None
    return policy


def run_many(policy, n=1000):
    results = [mdp.run_episode(policy) for _ in range(n)]
    outcomes = [r[2] for r in results]
    rewards = [r[1] for r in results]

    return {
        "goal": outcomes.count("goal") / n,
        "hazard": outcomes.count("hazard") / n,
        "timeout": outcomes.count("timeout") / n,
        "avg_reward": sum(rewards) / n,
    }


def print_policy(policy, title):
    print(title)
    for y in range(mdp.HEIGHT, 0, -1):
        row = []
        for x in range(1, mdp.WIDTH + 1):
            s = (x, y)
            if s == mdp.GOAL:
                row.append(" GOAL")
            elif s in mdp.TERMINALS and mdp.TERMINALS[s] < 0:
                row.append(" HAZD")
            elif s == mdp.WALL:
                row.append(" WALL")
            else:
                arrow = mdp.ARROWS[policy[s]]
                row.append(f"  {arrow}  ")
        print("".join(row))
    print()


def part3_compare_optimal_vs_greedy():
    print("Part 3: Comparison with Naive Greedy Policy")

    random.seed(42)
    V, _ = mdp.value_iteration(gamma=0.99)
    optimal_policy = mdp.extract_policy(V, gamma=0.99)
    greedy_policy = build_greedy_policy()

    opt = run_many(optimal_policy, n=1000)
    greedy = run_many(greedy_policy, n=1000)

    print("Optimal policy:")
    print(f"  Goal reached:  {opt['goal']:.3f}")
    print(f"  Hazard hit:    {opt['hazard']:.3f}")
    print(f"  Avg reward:    {opt['avg_reward']:.3f}")

    print("\nGreedy policy:")
    print(f"  Goal reached:  {greedy['goal']:.3f}")
    print(f"  Hazard hit:    {greedy['hazard']:.3f}")
    print(f"  Avg reward:    {greedy['avg_reward']:.3f}")
    print()


def part4_discount_experiment():
    print("Part 4: Discount Factor Experiment")
    random.seed(42)

    for gamma in [0.1, 0.5, 0.9, 0.99]:
        V, _ = mdp.value_iteration(gamma=gamma)
        policy = mdp.extract_policy(V, gamma=gamma)
        stats = run_many(policy, n=1000)
        print(
            f"gamma={gamma:.2f}  goal={stats['goal']:.3f}  "
            f"hazard={stats['hazard']:.3f}  avg_reward={stats['avg_reward']:.3f}"
        )
    print()


def part5_harder_warehouse():
    print("Part 5: Harder Warehouse (add hazard at (2,3))")

    random.seed(42)

    # Temporarily add second hazard.
    original_terminals = dict(mdp.TERMINALS)
    try:
        mdp.TERMINALS[(2, 3)] = -1.0

        V, _ = mdp.value_iteration(gamma=0.99)
        policy = mdp.extract_policy(V, gamma=0.99)

        print_policy(policy, "Policy with second hazard at (2,3):")

        stats = run_many(policy, n=1000)
        print(f"Goal reached:  {stats['goal']:.3f}")
        print(f"Hazard hit:    {stats['hazard']:.3f}")
        print(f"Avg reward:    {stats['avg_reward']:.3f}")
        print()
    finally:
        mdp.TERMINALS.clear()
        mdp.TERMINALS.update(original_terminals)


if __name__ == "__main__":
    part3_compare_optimal_vs_greedy()
    part4_discount_experiment()
    part5_harder_warehouse()
