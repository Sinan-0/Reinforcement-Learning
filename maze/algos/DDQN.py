import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import numpy as np
import random
from collections import deque

import time

action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

class BasicBuffer:
    '''
    Store experiences
    '''
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, np.array([action]), np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        
        batch = random.sample(self.buffer, batch_size)
        
        for experience in batch:
            state, action, reward, next_state, done = experience
            #print(experience)
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)


class ConvDQN(nn.Module):
    '''
    DQN using Convolutional layers
    '''
    
    def __init__(self, input_dim, output_dim):
        super(ConvDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.feature_size()
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        features = self.conv_net(state)
        features = features.view(features.size(0), -1)
        qvals = self.fc(features)
        return qvals

    def feature_size(self):
        return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


class DQN(nn.Module):
    '''
    DQN using Linear layers
    '''
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim[0], 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )
        
        self.fc.apply(self.init_weights)
        
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight.data)
    

    def forward(self, state):
        qvals = self.fc(state)
        return qvals

class DQNAgent:
    '''
    Agent
    '''
    def __init__(self, env, use_conv=False, learning_rate=1e-4, gamma=0.99, tau = 0.01, buffer_size=10000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
	
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        
        self.use_conv = use_conv
        if self.use_conv:
            self.model = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
            self.target_model = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        else:
            self.model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
            self.target_model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)

        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)
    
        print(self.model)
        print(self.target_model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def get_action(self, state, eps=0.1):
        stttt = state
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        
        #if stttt[0] == 1 and stttt[1] == 1:
            #print(state.detach().numpy(), qvals.detach().numpy(), action)
        
        if(np.random.randn() < eps):
            return self.env.action_space.sample()

        return action

    def compute_loss(self, batch): 
        #get data from created batch
        states, actions, rewards, next_states, dones = batch
        
        #Convert to torch.tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)
        
        # resize tensors
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        # compute loss
        curr_Q = self.model.forward(states).gather(1, actions)
        next_Q = self.target_model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q
        
        loss = F.mse_loss(curr_Q, expected_Q.detach())
        
        return loss
        

    def update(self, batch_size):
        #optimize the model
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)
        #print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        # target network update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
    
    
    
class Clipped_DQNAgent:
    '''
    Same as previous agent, except that we use 2 models 
    '''

    def __init__(self, env, use_conv=True, learning_rate=1e-7, gamma=0.99, tau=0.01, buffer_size=10000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
	
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(env.observation_space.shape)
        self.use_conv = use_conv
        if self.use_conv:
            self.model1 = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
            self.model2 = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        else:
            self.model1 = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
            self.model2 = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
    
    
        print(self.model1)
        print(self.model2)
        
        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=learning_rate)
        self.optimizer2 = torch.optim.Adam(self.model2.parameters(), lr=learning_rate)
        
    def get_action(self, state, eps=0.05):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model1.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        
        if(np.random.randn() < eps):
            return self.env.action_space.sample()

        return action

    def compute_loss(self, batch):     
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        # resize tensors
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)
        #print(dones)

        # compute loss
        curr_Q1 = self.model1.forward(states).gather(1, actions)
        curr_Q2 = self.model2.forward(states).gather(1, actions)
        
        next_Q1 = self.model1.forward(next_states)
        next_Q2 = self.model2.forward(next_states)
        next_Q = torch.min(
            torch.max(self.model1.forward(next_states), 1)[0],
            torch.max(self.model2.forward(next_states), 1)[0]
        )
        next_Q = next_Q.view(next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * next_Q
        
        loss1 = F.mse_loss(curr_Q1, expected_Q.detach())
        loss2 = F.mse_loss(curr_Q2, expected_Q.detach())

        return loss1, loss2

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        #(state_batch, action_batch, reward_batch, next_state_batch, done_batch) = batch
        #print(state_batch, '\n', action_batch, '\n', reward_batch, '\n', next_state_batch, '\n', done_batch, '\n')
        loss1, loss2 = self.compute_loss(batch)
        #print(loss1, loss2)
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()
        '''
class DQNAgent:

    def __init__(self, env, use_conv=True, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
	
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_conv = use_conv
        if self.use_conv:
            self.model = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        else:
            self.model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, eps=0.20):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        
        if(np.random.randn() < eps):
            return self.env.action_space.sample()

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        '''
def run_DDQN(env, agent, max_episodes, max_steps, batch_size, verbose = True):
    def tuple_to_int(SIZE, _tuple):
        """Convert coordinqates into unique key --> a state will be an integer that represents a coordinate
        see https://math.stackexchange.com/questions/1588601/create-unique-identifier-from-close-coordinates
        """
        return _tuple[1] + _tuple[0]*SIZE
    
    nb_steps_to_finish = []
    all_state_visits = [] #contains number of visits per state for last 10 episode
    for episode in range(max_episodes):
        obs = env.reset() #reset environment
        state = tuple_to_int(env.size, obs)  #convert coordinate into integer
        state_visits = np.zeros(env.size*env.size) #contains number of visits per state
        state_visits[state] +=1 #add the visit for the starting point
        nb_step = 0

        for step in range(max_steps):
            action = agent.get_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            
            
            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

                
            #show evolution
            env.render(mode='human')
            
            # update for next iteration
            next_state = tuple_to_int(env.size, next_obs) #convert new observation into a state
            state = next_state
            obs = next_obs
            nb_step+=1
            
            #add the visits per state for last 10 episodes
            if episode+1 >= max_episodes-10:
                state_visits[state] +=1
                
            
            #time.sleep(5)
            
            if done or step == max_steps-1:
                break
            
        #add result info
        if episode >= max_episodes-10:
            all_state_visits.append(state_visits)
        nb_steps_to_finish.append(nb_step)

        if verbose:
            print('Episode {}  | Nb Steps to finish: {}'.format(episode + 1, nb_step))

    return nb_steps_to_finish, all_state_visits

## ORGINAL USAGE
'''
env_id = "CartPole-v0"
MAX_EPISODES = 1000
MAX_STEPS = 500
BATCH_SIZE = 32

env = gym.make(env_id)
agent = DQNAgent(env, use_conv=False)
episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)'''
