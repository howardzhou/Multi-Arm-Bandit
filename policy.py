import numpy as np
import time

class Policy(object):
    """
    A policy prescribes an action to be taken based on the memory of an agent.
    """
    def __str__(self):
        return 'generic policy'

    def choose(self, agent):
        return 0


class EpsilonGreedyPolicy(Policy):
    """
    The Epsilon-Greedy policy will choose a random action with probability
    epsilon and take the best apparent approach with probability 1-epsilon. If
    multiple actions are tied for best choice, then a random action from that
    subset is selected.
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return '\u03B5-greedy (\u03B5={})'.format(self.epsilon)

    def choose(self, agent):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(agent.value_estimates))
        else:
            action = np.argmax(agent.value_estimates)
            check = np.where(agent.value_estimates == action)[0]
            if len(check) == 0:
                return action
            else:
                return np.random.choice(check)


class GreedyPolicy(EpsilonGreedyPolicy):
    """
    The Greedy policy only takes the best apparent action, with ties broken by
    random selection. This can be seen as a special case of EpsilonGreedy where
    epsilon = 0 i.e. always exploit.
    """
    def __init__(self):
        super(GreedyPolicy, self).__init__(0)

    def __str__(self):
        return 'greedy'


class RandomPolicy(EpsilonGreedyPolicy):
    """
    The Random policy randomly selects from all available actions with no
    consideration to which is apparently best. This can be seen as a special
    case of EpsilonGreedy where epsilon = 1 i.e. always explore.
    """
    def __init__(self):
        super(RandomPolicy, self).__init__(1)

    def __str__(self):
        return 'random'


class UCBPolicy(Policy):
    """
    The Upper Confidence Bound algorithm (UCB1). It applies an exploration
    factor to the expected value of each arm which can influence a greedy
    selection strategy to more intelligently explore less confident options.
    """
    def __init__(self, c):
        self.c = c

    def __str__(self):
        return 'UCB (c={})'.format(self.c)

    def choose(self, agent):
        exploration = np.log(agent.t+1) / agent.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1/self.c)

        q = agent.value_estimates + exploration
        action = np.argmax(q)
        check = np.where(q == action)[0]
        if len(check) == 0:
            return action
        else:
            return np.random.choice(check)


class SoftmaxPolicy(Policy):
    """
    The Softmax policy converts the estimated arm rewards into probabilities
    then randomly samples from the resultant distribution. This policy is
    primarily employed by the Gradient Agent for learning relative preferences.
    """
    def __str__(self):
        return 'SM'

    def choose(self, agent):
        a = agent.value_estimates
        pi = np.exp(a) / np.sum(np.exp(a))
        cdf = np.cumsum(pi)
        s = np.random.random()
        return np.where(s < cdf)[0][0]

class GaussianUCBPolicy(Policy):    # By Howard
    """
    The Gaussian UCB policy uses the method mentioned on class, where both 
    the prior and observations are Gaussian.
    """
    def __init__(self, c):
        self.c = c

    def __str__(self):
        return 'Gaussian UCB (c={})'.format(self.c)

    def choose(self, agent):
        # Update prior variance
        Pmtx = np.diag(agent.action_attempts / agent.obsVar)    # P^{-1}
        _Gamma = np.linalg.inv(agent.priorCov) + Pmtx
        _Covt = np.linalg.inv(_Gamma)

        # Calculate metrix
        maxAgtIdx = 0
        maxAgtVlue = 0.0
        _varVec = np.power(np.diag(_Covt),0.5)
        _covMtx = np.outer(_varVec,_varVec)
        _rho = np.power(_Covt / _covMtx,2)
        _sumRho = np.power(_rho.sum(axis=1),0.5)
        # print(_sumRho)

        # Howard: here assume the alpha=0 or 1 or infty
        exploration = np.multiply(np.power(np.diag(_Covt),0.5), _sumRho)
        exploration[np.isnan(exploration)] = 0
        # print(exploration)

        # Update estimate
        _Pm = np.dot(Pmtx,np.transpose(agent._value_estimates))
        _test1 = np.transpose(agent.priorMean)
        _mu = np.dot(_Covt,(_Pm+ np.dot(np.linalg.inv(agent.priorCov),np.transpose(agent.priorMean))))
        q = _mu + exploration
        # print(agent.obsCum )
        # time.sleep(2)
        # print(_mu)
        # print(_mu.size)
        action = np.argmax(q)
        # print(q)
        # print(exploration)
        # print(action)
        # time.sleep(0.5)
        check = np.where(q == action)[0]
        check = np.where(agent.action_attempts == 0)[0]
        # print(check)
        # time.sleep(2)
        if len(check) == 0:
            return action
        else:
            return np.random.choice(check)
        # print("test")
