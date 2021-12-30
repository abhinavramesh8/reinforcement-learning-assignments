import numpy as np

EPS = 0.1
N_ARMS = 10
NOISE_MEAN = 0
NOISE_DEV  = 0.1
REWARD_DEV = 1
ALPHA = 0.1
N_ITERS = 10000
N_RUNS = 300

class Bandit:
    def __init__(self, average_type):
        self.true_expectations = np.zeros(N_ARMS)
        self.estimated_expectations = np.zeros(N_ARMS)
        self.counts = np.zeros(N_ARMS, dtype=int)
        self.rewards = np.zeros(N_ITERS)
        self.optimal_actions = np.zeros(N_ITERS, dtype=int)
        if average_type == 'sample':
            self.alpha = lambda : 1/self.counts[self.idx]
        else:
            self.alpha = lambda : ALPHA
    
    def run(self):
        for iteration in range(N_ITERS):
            reward = self.pull_arm()
            self.update_attrs(reward, iteration)
        return (self.rewards, self.optimal_actions)

    def pull_arm(self):
        if np.random.rand() > EPS:
            self.idx = np.argmax(self.estimated_expectations)
        else:
            self.idx = np.random.randint(N_ARMS)
        return np.random.normal(self.true_expectations[self.idx], REWARD_DEV)

    def update_attrs(self, reward, iteration):
        self.counts[self.idx] += 1
        self.estimated_expectations[self.idx] += \
            (self.alpha() * (reward - self.estimated_expectations[self.idx]))
        self.rewards[iteration] = reward
        if np.argmax(self.true_expectations) == self.idx:
            self.optimal_actions[iteration] = 1
        self.true_expectations += \
            np.random.normal(NOISE_MEAN, NOISE_DEV, N_ARMS)

def avg_rewards_optimal_actions(average_type):
    avg_rewards = np.zeros(N_ITERS)
    avg_optimal_actions = np.zeros(N_ITERS, dtype=int)
    for _ in range(N_RUNS):
        rewards, optimal_actions = Bandit(average_type).run()
        avg_rewards += rewards
        avg_optimal_actions += optimal_actions
    avg_rewards /= N_RUNS
    avg_optimal_actions = avg_optimal_actions.astype(float) / N_RUNS
    return (avg_rewards, avg_optimal_actions)

sample_rewards, sample_optimal = avg_rewards_optimal_actions('sample')
weighted_rewards, weighted_optimal = avg_rewards_optimal_actions('weighted')
result = (sample_rewards, sample_optimal, weighted_rewards, weighted_optimal)
np.savetxt("result.out", result, fmt="%.3f")