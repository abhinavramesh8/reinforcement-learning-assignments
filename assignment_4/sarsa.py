import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        tiling_shape = np.ceil((state_high - state_low) / tile_width).astype(int) + 1
        self.feature_vector_shape = (num_actions, num_tilings, *tiling_shape)
        self.n_features = np.prod(self.feature_vector_shape)

        offset = np.outer(np.arange(num_tilings), tile_width) / num_tilings
        self.start_index = state_low - offset

        self.n_tilings = num_tilings
        self.tile_width = tile_width

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.n_features

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        x = np.zeros(self.feature_vector_shape, dtype=int)
        if not done:
            active_idx = ((s - self.start_index) / self.tile_width).astype(int)
            for tiling_num, idx in enumerate(active_idx):
                x[(a, tiling_num, *idx)] = 1   
        return x.flatten()
        

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros(X.feature_vector_len())

    for _ in range(num_episode):
        s, done = env.reset(), False
        a = epsilon_greedy_policy(s, done, w)
        x = X(s, done, a)
        z = np.zeros(w.shape)
        q_old = 0
        while not done:
            s_prime, r, done, _ = env.step(a)
            a_prime = epsilon_greedy_policy(s_prime, done, w)
            x_prime = X(s_prime, done, a_prime)
            q = w.dot(x)
            q_prime = w.dot(x_prime)
            delta = r + gamma * q_prime - q
            z = gamma * lam * z + (1 - alpha * gamma * lam * z.dot(x)) * x
            w += alpha * ((delta + q - q_old) * z - (q - q_old) * x)
            q_old = q_prime
            x = x_prime
            a = a_prime

    return w
