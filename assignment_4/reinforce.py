from typing import Iterable
import numpy as np
import torch
from torch.distributions.categorical import Categorical

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size
        n_hidden = 32
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dims, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, num_actions),
            torch.nn.Softmax(dim=1)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)

    def __call__(self,s) -> int:
        with torch.no_grad():
            self.model.eval()
            input = torch.as_tensor(s, dtype=torch.float32)[None]
            action_probs = self.model(input)[0]
            distribution = Categorical(probs=action_probs)
            return distribution.sample().item()

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.model.train()
        input = torch.as_tensor(s, dtype=torch.float32)[None]
        output = self.model(input)
        loss_val = - gamma_t * delta * output[0, a].log()
        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()


class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass


class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        n_hidden = 32
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dims, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, 1)
        )
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)

    def __call__(self,s) -> float:
        with torch.no_grad():
            self.model.eval()
            input = torch.as_tensor(s, dtype=torch.float32)[None]
            output = self.model(input)
            return output.item()

    def update(self,s,G):
        self.model.train()
        input = torch.as_tensor(s, dtype=torch.float32)[None]
        out = self.model(input)
        truth = torch.as_tensor(G, dtype=torch.float32)[None, None]
        loss_val = self.loss(out, truth)
        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    G0_arr = []
    for dummy1 in range(num_episodes):
        R = [None]
        S = []
        A = []
        s, done = env.reset(), False
        while not done:
            S.append(s)
            a = pi(s)
            A.append(a)
            s, r, done, dummy2 = env.step(a)
            R.append(r)
        T = len(R) - 1
        for t in range(T):
            G = 0.0
            for k in range(t+1, T+1):
                G += gamma ** (k - t - 1) * R[k]
            if t == 0:
                G0_arr.append(G)
            V.update(S[t], G)
            delta = G - V(S[t])
            gamma_t = gamma ** t
            pi.update(S[t], A[t], gamma_t, delta)
    return G0_arr