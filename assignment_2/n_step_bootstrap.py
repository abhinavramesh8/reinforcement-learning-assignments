from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

class DeterministicPolicy(Policy):
    def __init__(self, nS):
        self.act = [0] * nS
    
    def set_action(self, s, a):
        self.act[s] = a
    
    def action_prob(self, s, a):
        return 1 if a == self.act[s] else 0
    
    def action(self, s):
        return self.act[s]

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    V = initV

    for traj in trajs:
        T = len(traj)
        for tau in range(0, T):
            G = 0.0
            for i in range(tau + 1, min(tau + n, T) + 1):
                G += env_spec.gamma ** (i - tau - 1) * traj[i-1][2]
            if tau + n < T:
                G += env_spec.gamma ** n * V[traj[tau + n][0]]
            
            s = traj[tau][0]
            V[s] += alpha * (G - V[s])

    return V

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    Q = initQ
    pi = DeterministicPolicy(env_spec.nS)
    for s in range(env_spec.nS):
        pi.set_action(s, np.argmax(Q[s]))
    
    for traj in trajs:
        T = len(traj)
        for tau in range(T):
            rho = 1
            for i in range(tau + 1, min(tau + n, T - 1) + 1):
                s, a = traj[i][:2]
                rho *= pi.action_prob(s, a) / bpi.action_prob(s, a)

            G = 0.0
            for i in range(tau + 1, min(tau + n, T) + 1):
                G += env_spec.gamma ** (i - tau - 1) * traj[i-1][2]
            if tau + n < T:
                s, a = traj[tau + n][:2]
                G += env_spec.gamma ** n * Q[s][a]
            
            s, a = traj[tau][:2]
            Q[s][a] += alpha * rho * (G - Q[s][a])
            pi.set_action(s, np.argmax(Q[s]))

    return Q, pi
