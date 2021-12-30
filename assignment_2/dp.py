from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

class DeterministicPolicy(Policy):
    def __init__(self, nS):
        self.act = list(range(nS))
    
    def set_action(self, s, a):
        self.act[s] = a
    
    def action_prob(self, s, a):
        return 1 if a == self.act[s] else 0
    
    def action(self, s):
        return self.act[s]
        
def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    V = initV
    delta = float('inf')
    nS = env.spec.nS
    nA = env.spec.nA
    gamma = env.spec.gamma

    while delta >= theta:
        delta = 0.0

        for s in range(nS):
            v = V[s]
            s_value = 0.0

            for a in range(nA):
                a_reward = 0.0

                for ss in range(nS):
                    ss_reward = env.R[s][a][ss] + gamma * V[ss]
                    a_reward += env.TD[s][a][ss] * ss_reward

                s_value += pi.action_prob(s, a) * a_reward

            V[s] = s_value
            delta = max(delta, abs(v - V[s]))

    Q = np.zeros(shape=(nS, nA))

    for s in range(nS):
        for a in range(nA):
            for ss in range(nS):
                ss_reward = env.R[s][a][ss] + gamma * V[ss]
                Q[s][a] += env.TD[s][a][ss] * ss_reward

    return V, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    V = initV
    delta = float('inf')
    nS = env.spec.nS
    nA = env.spec.nA
    gamma = env.spec.gamma

    while delta >= theta:
        delta = 0.0

        for s in range(nS):
            v = V[s]
            action_rewards = []

            for a in range(nA):
                a_reward = 0.0

                for ss in range(nS):
                    ss_reward = env.R[s][a][ss] + gamma * V[ss]
                    a_reward += env.TD[s][a][ss] * ss_reward

                action_rewards.append(a_reward)

            V[s] = max(action_rewards)
            delta = max(delta, abs(v - V[s]))
    
    pi = DeterministicPolicy(nS)

    for s in range(nS):
        action_rewards = []

        for a in range(nA):
            a_reward = 0.0

            for ss in range(nS):
                ss_reward = env.R[s][a][ss] + gamma * V[ss]
                a_reward += env.TD[s][a][ss] * ss_reward

            action_rewards.append(a_reward)
        
        pi.set_action(s, np.argmax(action_rewards))

    return V, pi
