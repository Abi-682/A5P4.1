"""Compatibility entrypoint: same deliverable as warehouse_mdp_agent.py."""

from warehouse_mdp_agent import *  # noqa: F401,F403
from warehouse_mdp_agent import _run_optimal_policy_demo, _verify_simulate_step


if __name__ == "__main__":
    _verify_simulate_step()
    _run_optimal_policy_demo()
