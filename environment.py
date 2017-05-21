import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

from bandits.agent import BetaAgent


class Environment(object):
    def __init__(self, bandit, agents, label='Multi-Armed Bandit'):
        self.bandit = bandit
        self.agents = agents
        self.label = label

    def reset(self):
        self.bandit.reset()
        for agent in self.agents:
            agent.reset()

    def initial_prior(self,priorMean=0,priorCov=0):
        for agent in self.agents:
            agent.initial_prior(priorMean,priorCov)


    def run(self, trials=100, experiments=1):
        scores = np.zeros((trials, len(self.agents)))
        optimal = np.zeros_like(scores)
        regrets = np.zeros((trials, len(self.agents)))

        for _ in range(experiments):
            # priorMean = [i for i in range(1,11)]    # Howard: prior mean
            priorMean = np.zeros(10,float)
            priorCov = np.zeros((10,10),float)
            np.fill_diagonal(priorCov,1000)                   # Howard: prior covariance matrix
            self.initial_prior(priorMean,priorCov)
            self.reset()
            for t in range(trials):
                for i, agent in enumerate(self.agents):
                    action = agent.choose()
                    reward, is_optimal, this_reg = self.bandit.pull(action)
                    agent.observe(reward)
                    regrets[t, i] += this_reg

                    scores[t, i] += reward
                    if is_optimal:
                        optimal[t, i] += 1

        return scores / experiments, optimal / experiments, regrets

    def plot_results(self, scores, optimal, regrets):
        sns.set_style('white')
        sns.set_context('talk')
        plt.subplot(3, 1, 1)
        plt.title(self.label)
        plt.plot(scores)
        plt.ylabel('Average Reward')
        plt.legend(self.agents, loc=4)
        plt.subplot(3, 1, 2)
        plt.plot(optimal * 100)
        plt.ylim(0, 100)
        plt.ylabel('% Optimal Action')
        plt.xlabel('Time Step')
        plt.legend(self.agents, loc=4)
        plt.subplot(3, 1, 3)
        plt.plot(regrets)
        plt.ylabel('Cum. Regrets')
        plt.legend(self.agents, loc=4)
        sns.despine()
        plt.show()

    def plot_beliefs(self):
        sns.set_context('talk')
        pal = sns.color_palette("cubehelix", n_colors=len(self.agents))
        plt.title(self.label + ' - Agent Beliefs')

        rows = 2
        cols = int(self.bandit.k / 2)

        axes = [plt.subplot(rows, cols, i+1) for i in range(self.bandit.k)]
        for i, val in enumerate(self.bandit.action_values):
            color = 'r' if i == self.bandit.optimal else 'k'
            axes[i].vlines(val, 0, 1, colors=color)

        for i, agent in enumerate(self.agents):
            if type(agent) is not BetaAgent:
                for j, val in enumerate(agent.value_estimates):
                    axes[j].vlines(val, 0, 0.75, colors=pal[i], alpha=0.8)
            else:
                x = np.arange(0, 1, 0.001)
                y = np.array([stats.beta.pdf(x, a, b) for a, b in
                             zip(agent.alpha, agent.beta)])
                y /= np.max(y)
                for j, _y in enumerate(y):
                    axes[j].plot(x, _y, color=pal[i], alpha=0.8)

        min_p = np.argmin(self.bandit.action_values)
        for i, ax in enumerate(axes):
            ax.set_xlim(0, 1)
            if i % cols != 0:
                ax.set_yticklabels([])
            if i < cols:
                ax.set_xticklabels([])
            else:
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(['0', '', '0.5', '', '1'])
            if i == int(cols/2):
                title = '{}-arm Bandit - Agent Estimators'.format(self.bandit.k)
                ax.set_title(title)
            if i == min_p:
                ax.legend(self.agents)

        sns.despine()
        plt.show()
