import random
from memory import PPOMemory
import numpy as np
import torch
import gym


from neuralnet import SeaquestNet
from collections import deque

MAX_MEMORY = 10000
BATCH_SIZE = 32
ACTIONSPACE = 18
MEMORY_SIZE = 32

class Agent:
    def __init__(self, env,  epsilon, gamma, learning_rate, trainer_class, batch_size=8):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.net = SeaquestNet(learning_rate)
        self.memory = PPOMemory(batch_size, MEMORY_SIZE)
        self.trainer = trainer_class(self.net)

    def store_experience(self, state, action, probs, val, reward, done):
        self.memory.store_experience(state, action, probs, val, reward, done )


    def get_action(self, state):
        probas_torch = self.net.policy(torch.from_numpy(state))
        probas = probas_torch.detach().numpy()[0] # tensor to 1d np array
        action = np.random.choice(range(ACTIONSPACE), p=probas)
        val = self.net.stateValue(state)
        val = torch.squeeze(val).item()
        return probas, action, val

    def do_epoch(self):
        # for 1 epoch
        state = self.env.reset()
        score = 0
        done = False
        steps = 0

        while not done:
            probas, action, val = self.get_action(state)
            #perform step
            next_state, reward, done, _ = self.env.step((action))
            steps += 1
            score += reward
            self.store_experience(state, action, probas, val, reward, done)
            if steps % self.memory.memory_size == 0:
                self.trainer.train(self.memory)
            state = next_state

        self.save_model()
        return score




    def save_model(self):
        self.net.save_checkpoint()

    def load_model(self):
        self.net.load_checkpoint()

class PPOTrainer:
    GAMMA = 0.9
    LAMBDA = 0.95
    EPOCHS = 8
    CLIP = 0.2
    def __init__(self, net):
        self.net = net


    def ppo_loss(self, advantages, states, old_probs, actions ):
        CLIPVALUE = 0.2


    def advantages(self, rewards, values):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        delta_t = 0.0
        for t in range(len(rewards)-2,-1,-1): # index 18 -> 0
            factor = (self.GAMMA*self.LAMBDA)
            delta_t *= factor
            delta_t += rewards[t] + self.GAMMA*values[t+1]-values[t]
            advantages[t] = delta_t
        return advantages




    def train(self, memory):
        for epoch in range(self.EPOCHS):
            states, actions, probas, values, rewards, dones, batches = memory.generate_batches()
            advantages = self.advantages(rewards, values)

            advantages = torch.from_numpy(advantages) #make tensor

            for batch in batches:
                states = torch.from_numpy(states[batch])
                old_probas = torch.from_numpy(probas[batch])
                actions = torch.from_numpy(actions[batch]) # needed for log_probs

                new_probas = self.net.policy(states)
                critic_values = self.net.stateValue(states)

                new_log_probas = new_probas.log_prob(actions)
                old_log_probas = old_probas.log_prob(actions) #moest log_prob al eerder? not sure

                prob_ratios = new_log_probas.exp() / old_log_probas.exp()
                weighted_probs = advantages[batch] * prob_ratios
                clipped_weighted_probs = torch.clamp(prob_ratios, 1-self.CLIP, 1+self.CLIP)*advantages[batch]
                actor_loss = -torch.min(weighted_probs, clipped_weighted_probs).mean()
                #mask value -> dones is 0's or 1's????????? for Advantages

                returns = advantages[batch] + values[batch] # waarom dit???????????
                critic_loss = (returns-critic_values)**2















