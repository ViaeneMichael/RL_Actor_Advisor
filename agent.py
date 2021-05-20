import random
from memory import PPOMemory
import numpy as np
import torch
import torch.autograd as autograd

import gym
from scipy.stats import entropy


from neuralnet import SeaquestNet
from collections import deque

MAX_MEMORY = 10000
BATCH_SIZE = 32
ACTIONSPACE = 18
MEMORY_SIZE = 12

#Cuda
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class Agent:
    def __init__(self, env,  epsilon, gamma, learning_rate, trainer_class, batch_size=8):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.net = SeaquestNet(learning_rate, env.observation_space.shape)
        self.net.load_checkpoint()

        self.memory = PPOMemory(batch_size, MEMORY_SIZE)
        self.trainer = trainer_class(self.net)

    def store_experience(self, state, action, probs, val, reward, done):
        self.memory.store_experience(state, action, probs, val, reward, done )

    def to_ram(self, ale):
        ram_size = ale.getRAMSize()
        ram = np.zeros((ram_size), dtype=np.uint8)
        ale.getRAM(ram)
        return ram

    def advice(self, probas_torch):
        ram = self.to_ram(self.env.ale)
        depth_enc = map(int, list(bin(ram[97] - 13)[2:].rjust(7, '0')))
        oxy_full = 1 if ram[102] == 64 else 0
        oxy_low = 1 if ram[102] <= 4 else 0
        diver_found = 1 if ram[62] > 0 else 0
        # ups: 2,6,7,10,14,15
        # downs: 5,8,9,13,16,17
        # depth: surface level = 13
        #depth bottom =

        #rules: als 6 divers, ups
        #rules: no divers and enough oxygen , not higher than 20 depth
        #rules: oxy low, ups

    def get_action(self, state):
        probas_torch = self.net.policy(state)
        self.advice(probas_torch)
        action = probas_torch.sample()
        proba = torch.squeeze(probas_torch.log_prob(action)).item()
        action = torch.squeeze(action).item()


        val = self.net.stateValue(state)
        val = torch.squeeze(val).item()
        return proba, action, val

    def do_episode(self):
        # for 1 epoch
        state = self.env.reset()
        score = 0
        done = False
        steps = 0

        while not done:
            # state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            proba, action, val = self.get_action(Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True))
            #perform step
            next_state, reward, done, _ = self.env.step(action)
            steps += 1
            score += reward
            self.store_experience(state, action, proba, val, reward, done)
            if steps % self.memory.memory_size == 0:
                self.trainer.train(self.memory)
            state = next_state
        if len(self.memory.states) > 0:
            self.trainer.train(self.memory) # memory is cleared in train
        self.save_model()
        return score


    def save_model(self):
        self.net.save_checkpoint()

    def load_model(self):
        self.net.load_checkpoint()

class PPOTrainer:
    GAMMA = 0.9
    LAMBDA = 0.95
    EPOCHS = 1
    CLIP = 0.2
    def __init__(self, net):
        self.net = net


    def ppo_loss(self, advantages, states, old_probs, actions ):
        CLIPVALUE = 0.2


    def advantages(self, rewards, values, dones):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        delta_t = 0.0
        for t in range(len(rewards)-2,-1,-1): # index 18 -> 0
            mask = 0 if dones[t] else 1
            factor = (self.GAMMA*self.LAMBDA)
            delta_t *= factor
            delta_t = mask * delta_t + rewards[t] + mask * self.GAMMA*values[t+1]-values[t]
            advantages[t] = delta_t
        return advantages

    def train(self, memory):
        C1 = 0.5
        C2 = 0.5
        for epoch in range(self.EPOCHS):
            states, actions, probas, values, rewards, dones = memory.generate_batches()
            advantages = self.advantages(rewards, values, dones)

            advantages = torch.from_numpy(advantages) #make tensor
            p = np.random.permutation(len(states))
            states = states[p]
            actions = actions[p]
            probas = probas[p]
            values = values[p]
            advantages = advantages[p]
            states_batch = torch.FloatTensor(states)
            old_probas = torch.from_numpy(probas)
            actions_batch = torch.from_numpy(actions)  # needed for log_probs

            new_probas = self.net.policy(states_batch)
            critic_values = self.net.stateValue(states_batch)

            critic_values = torch.squeeze(critic_values)

            new_log_probas = new_probas.log_prob(actions_batch)

            prob_ratios = new_log_probas.exp() / old_probas.exp()
            weighted_probs = advantages * prob_ratios
            clipped_weighted_probs = torch.clamp(prob_ratios, 1-self.CLIP, 1+self.CLIP)*advantages
            actor_loss = -torch.min(weighted_probs, clipped_weighted_probs).mean()

            returns = advantages + values
            critic_loss = ((returns-critic_values)**2)
            critic_loss = critic_loss.mean()

            entropy_loss = new_probas.entropy().mean()

            total_loss = actor_loss + (C1 * critic_loss) - (C2 * entropy_loss)
            self.net.optimizer.zero_grad()
            total_loss.backward()
            self.net.optimizer.step()

        memory.clear()

















