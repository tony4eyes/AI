#AI 2018
import os
import numpy as np

#hyper parameters

class Hp():
    def __init__(self):
        self.nb_steps = 1000
        self.episode_length = 1000
        self.learning_rate = 0.02
        self.nb_directions = 16
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.03
        self.seed = 1
        self.env_name = ''
        
#normalizing the states
        
class Normalize():
    def __init__(self, nb_inputs):     # num of inputs
        self.n = np.zeros(nb_inputs)   # num of elements in input vector
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)
        
    def observe(self, x):
        self.n += 1.;                  # num of state, new state
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)  #np.clip bound the lower to be 1e-2
        
    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

#building AI  (policy)
        
class Policy():
    def __init__(self, input_size, output_size)：:
        #output_size: num of action to play
        #perceptron: the algo: one layer neural network, matrix of weight Theta
        self.theta = np.zeros((output_size, input_size))
        
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return (self.theta.dot(input))
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input)
        else:
            return (self.theta - hp.noise*delta).dot(input)
        
    def sample_deltas(self):
        return [np.randn(*self.theta.shape) for _ in range(hp.nb_directions)]
    
    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos,r_neg, d in (rollouts):
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step
        
#Exporing the policy on one specific direction and over one episode
def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    num_plays = 0;
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        #outlier reward, clip them between -1 and 1
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards

def train(env, policy, normalizer, hp):
    for step in range(hp.nb_steps):
        #Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions