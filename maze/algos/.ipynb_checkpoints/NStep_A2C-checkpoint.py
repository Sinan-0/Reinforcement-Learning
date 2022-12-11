import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pandas as pd

##see https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
# and https://github.com/hermesdt/reinforcement-learning/blob/master/a2c/cartpole_a2c_episodic.ipynb



def t(x): return torch.from_numpy(x).float()


# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=0)
        )
        self.model.apply(self.init_weights)
        
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight.data)
    
    def forward(self, X):
        return self.model(X)
    
    
# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.model.apply(self.init_weights)
        
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight.data)
    
    def forward(self, X):
        return self.model(X)
    
    
class N_Step_A2CAgent():
    def __init__(self, env, learning_rate=1e-4, gamma=0.99, buffer_size=10000):
        self.env = env
        self.gamma = gamma
        
        # config
        self.state_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        
        self.actor = Actor(self.state_dim, self.n_actions)
        self.critic = Critic(self.state_dim)
        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        print(self.actor)
        print(self.critic)
        
    def model_forward(self, state):
        state = t(state)
        actor_output = self.actor(state)
        critic_output = self.critic(state)
        return actor_output, critic_output
    
    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * self.gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r
    
    def update(self, log_probs, rewards, values, last_q_val):
        
        # compute Q values
        Qvals = np.zeros((len(values), 1))
        for t in reversed(range(len(rewards))):
            last_q_val = rewards[t] + self.gamma * last_q_val
            Qvals[t] = last_q_val
            
        values = torch.stack(values)
        Qvals = torch.Tensor(Qvals)
        log_probs = torch.stack(log_probs)
        
        advantage = Qvals - values
        
        critic_loss = 0.5 * advantage.pow(2).mean()
        self.adam_critic.zero_grad()
        critic_loss.backward()
        self.adam_critic.step()

        actor_loss = (-log_probs * advantage.detach()).mean()
        self.adam_actor.zero_grad()
        actor_loss.backward()
        self.adam_actor.step()
        
def run_nstep_a2c(env, agent, nb_eps, max_steps, verbose = True):
    def tuple_to_int(SIZE, _tuple):
        """Convert coordinqates into unique key --> a state will be an integer that represents a coordinate
        see https://math.stackexchange.com/questions/1588601/create-unique-identifier-from-close-coordinates
        """
        return _tuple[1] + _tuple[0]*SIZE
    
    nb_steps_to_finish = []
    all_state_visits = [] #contains number of visits per state for last 10 episodes
    for episode in range(nb_eps):
        log_probs = []
        values = []
        rewards = []
        
        obs = env.reset() #reset environment
        #print(obs.shape)
        nb_step = 0
        done = False
        
        state = tuple_to_int(env.size, obs)  #convert coordinate into integer
        state_visits = np.zeros(env.size*env.size) #contains number of visits per state
        state_visits[state] +=1 #add the visit for the starting point
        while (not done) and (nb_step < max_steps):
            #compute actor critic outputs
            actor_output, critic_output = agent.model_forward(obs)
            
            #get action
            dist = Categorical(probs=actor_output)
            action = dist.sample()
            
            action_np = np.max(action.detach().numpy()) #torch.numpy() returns a weird version, that the action_to_text dictionnary in the env don't like, we thus convert the action to a good version
            next_obs, reward, done, _ = env.step(action_np)
            
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            values.append(critic_output)
            
            #show evolution
            env.render(mode='human')
                
            # update for next iteration
            obs = next_obs
            nb_step+=1
            
            if done or (nb_step == max_steps):
                _ , last_q_val = agent.model_forward(next_obs)
                last_q_val = last_q_val.detach().numpy()
                agent.update(log_probs, rewards, values, last_q_val)

            #qdd the visits per state for last 10 episodes
            state = tuple_to_int(env.size, next_obs) #update state
            if episode+1 >= nb_eps-10:
                state_visits[state] +=1
            
            #time.sleep(0.25)
            
        #add result info
        if episode >= nb_eps-10:
            all_state_visits.append(state_visits)
        nb_steps_to_finish.append(nb_step)
        
        if verbose:
            print('Episode {}  | Nb Steps to finish: {}'.format(episode + 1, nb_step))

    return agent, nb_steps_to_finish, all_state_visits