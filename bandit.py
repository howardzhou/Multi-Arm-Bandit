import numpy as np
import pymc3 as pm


class MultiArmedBandit(object):
    """
    A Multi-armed Bandit
    """
    def __init__(self, k):
        self.k = k
        self.action_values = np.zeros(k)
        self.optimal = 0

    def reset(self):
        self.action_values = np.zeros(self.k)
        self.optimal = 0

    def pull(self, action):
        return 0, True


class GaussianBandit(MultiArmedBandit):
    """
    Gaussian bandits model the reward of a given arm as normal distribution with
    provided mean and standard deviation.
    """
    def __init__(self, k, mu=0, sigma=1):  
        super(GaussianBandit, self).__init__(k)
        # self.mu = mu
        self.mu = np.random.normal(mu,1,self.k)
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.action_values = np.random.normal(self.mu, self.sigma, self.k)
        self.optimal = np.argmax(self.action_values)
        self.optimal_value = self.action_values[self.optimal]
        # print(self.optimal)

    def pull(self, action):
        # print(action)
        return (np.random.normal(self.action_values[action]),
                action == self.optimal,
                self.optimal_value-self.action_values[action])

