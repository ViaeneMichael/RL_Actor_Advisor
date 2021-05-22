from memory import PPOMemory
import numpy as np
import torch
import torch.autograd as autograd
from torch.distributions.categorical import Categorical
from neuralnet import SeaquestNet


MAX_MEMORY = 10000
ACTIONSPACE = 18
CLIP = 0.2

#Cuda
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class Agent:
    def __init__(self, env,  learning_rate, mem_size, gamma, lamb, c1,  trainer_class):
        self.env = env
        self.learning_rate = learning_rate

        self.net = SeaquestNet(learning_rate, env.observation_space.shape)

        # self.net.load_checkpoint()

        self.memory = PPOMemory(mem_size)
        self.trainer = trainer_class(self.net, gamma, lamb, c1, 4) # epochs hardcoded 4

    def store_experience(self, state, action, probs, val, reward, done):
        self.memory.store_experience(state, action, probs, val, reward, done )

    def to_ram(self, ale):
        ram_size = ale.getRAMSize()
        ram = np.zeros((ram_size), dtype=np.uint8)
        ale.getRAM(ram)
        return ram

    def advice(self):
        ram = self.to_ram(self.env.ale)
        depth_enc = ram[97]
        # oxy_full = 1 if ram[102] == 64 else 0
        oxy_low = 1 if ram[102] <= 4 else 0
        diver_found = 1 if ram[62] > 0 else 0
        advice = np.zeros(18)
        # ups: 2,6,7,10,14,15
        # downs: 5,8,9,13,16,17
        # depth: surface level = 13
        # rules: als 6 divers, ups
        # rules: oxy low, ups
        # rules: no divers and enough oxygen , not higher than 20 depth
        ups = [2,6,7,10,14,15]
        downs = [5,8,9,13,16,17]

        if ram[62] == 6 or oxy_low:
            advice[ups]=1.0/len(ups)
        elif not diver_found and not oxy_low and depth_enc<20:
            advice[downs]=1.0/len(downs)
        else:
            advice=np.ones(18)/len(advice)
        return advice

    #returns numpy array
    def combine_actor_and_advisor_policies(self, actor_probs, advisor_probs):
        normalization_factor = np.dot(actor_probs,advisor_probs)
        probas = np.multiply(actor_probs,advisor_probs)
        if normalization_factor == 0:
            print("Error")
        return probas/normalization_factor

    def get_action(self, state):
        actor_probs = self.net.policy(state)
        advisor_probs = self.advice()
        probas_torch = self.combine_actor_and_advisor_policies(actor_probs.detach().numpy(), advisor_probs)
        probas_torch = Categorical(torch.from_numpy(probas_torch))
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
    def __init__(self, net, gamma, lamb, c1, epochs):
        self.net = net
        self.GAMMA = gamma
        self.LAMBDA = lamb
        self.C1 = c1
        self.EPOCHS = epochs

    def advantages(self, rewards, values, dones):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        delta_t = 0.0
        for t in range(len(rewards)-2, -1, -1): # index 18 -> 0
            mask = 0 if dones[t] else 1
            factor = (self.GAMMA*self.LAMBDA)
            delta_t *= factor
            delta_t = mask * delta_t + rewards[t] + mask * self.GAMMA*values[t+1]-values[t]
            advantages[t] = delta_t
        return advantages

    def train(self, memory):
        C2 = 0.01
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

            new_probas = Categorical(self.net.policy(states_batch))
            critic_values = self.net.stateValue(states_batch)

            critic_values = torch.squeeze(critic_values)

            new_log_probas = new_probas.log_prob(actions_batch)

            prob_ratios = new_log_probas.exp() / old_probas.exp()
            weighted_probs = advantages * prob_ratios
            clipped_weighted_probs = torch.clamp(prob_ratios, 1-CLIP, 1+CLIP)*advantages
            actor_loss = -torch.min(weighted_probs, clipped_weighted_probs).mean()

            returns = advantages + values
            critic_loss = ((returns-critic_values)**2)
            critic_loss = critic_loss.mean()

            entropy_loss = new_probas.entropy().mean()

            total_loss = actor_loss + (self.C1 * critic_loss) - (C2 * entropy_loss)
            self.net.optimizer.zero_grad()
            total_loss.backward()
            self.net.optimizer.step()

        memory.clear()

















