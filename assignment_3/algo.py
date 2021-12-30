import numpy as np
from policy import Policy

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    for dummy1 in range(num_episode):
        s, done = env.reset(), False
        S = [s]
        R = [None]
        T = float('inf')
        t = 0
        while True:
            if t < T:
                a = pi.action(s)
                next_s, r, done, dummy2 = env.step(a)
                S.append(next_s)
                R.append(r)
                if done:
                    T = t + 1
            tau = t - n + 1
            if tau >= 0:
                G = 0.
                for i in range(tau+1, min(tau+n, T)+1):
                    G += gamma ** (i - tau - 1) * R[i]
                if tau + n < T:
                    G += gamma ** n * V(S[tau+n])
                V.update(alpha, G, S[tau])
            if tau == T - 1:
                break
            t += 1


