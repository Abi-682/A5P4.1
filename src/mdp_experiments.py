"""Compatibility entrypoint: same deliverable as warehouse_mdp_experiments.py."""

from warehouse_mdp_experiments import *  # noqa: F401,F403
from warehouse_mdp_experiments import (
    part3_compare_optimal_vs_greedy,
    part4_discount_experiment,
    part5_harder_warehouse,
)


if __name__ == "__main__":
    part3_compare_optimal_vs_greedy()
    part4_discount_experiment()
    part5_harder_warehouse()
