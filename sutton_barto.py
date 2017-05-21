"""
Multi-armed Bandit examples taken from Reinforcement Learning: An Introduction
by Sutton and Barto, 2nd ed. rev Oct2015.
"""
from bandits.environment import Environment
from bandits.bandit import GaussianBandit
from bandits.agent import Agent, GradientAgent, GaussianUCBAgent
from bandits.policy import (EpsilonGreedyPolicy, GreedyPolicy, UCBPolicy,
                            SoftmaxPolicy,GaussianUCBPolicy)
import numpy as np


class EpsilonGreedyExample:
    label = '2.2 - Action-Value Methods'
    bandit = GaussianBandit(10)
    agents = [
        Agent(bandit, GreedyPolicy()),
        Agent(bandit, EpsilonGreedyPolicy(0.01)),
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
    ]


class OptimisticInitialValueExample:
    label = '2.5 - Optimistic Initial Values'
    bandit = GaussianBandit(10)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
        Agent(bandit, GreedyPolicy(), prior=5)
    ]


class UCBExample:
    label = '2.6 - Upper-Confidence-Bound Action Selection'
    bandit = GaussianBandit(10)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
        Agent(bandit, UCBPolicy(2)),
        GaussianUCBAgent(bandit,GaussianUCBPolicy(2))
    ]


class GradientExample:
    label = '2.7 - Gradient Bandits'
    bandit = GaussianBandit(10, mu=4, sigma=10)
    policy = SoftmaxPolicy()
    agents = [
        GradientAgent(bandit, policy, alpha=0.1),
        GradientAgent(bandit, policy, alpha=0.4),
        GradientAgent(bandit, policy, alpha=0.1, baseline=False),
        GradientAgent(bandit, policy, alpha=0.4, baseline=False)
    ]


if __name__ == '__main__':
    experiments = 500
    trials = 1000

    SNR = 10
    


    # example = EpsilonGreedyExample
    # example = OptimisticInitialValueExample
    example = UCBExample
    # example = GradientExample

    env = Environment(example.bandit, example.agents, example.label)
    scores, optimal, regrets = env.run(trials, experiments)
    env.plot_results(scores, optimal,regrets)
