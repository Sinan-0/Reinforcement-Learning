import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import gym
from gym import spaces
#import pygame
import random

'''create maze'''
def create_maze(size, seed=None):
    #set the seed if one has bee
    if not(seed==None):
        np.random.seed(seed)
    #define starting and target point
    start_location=np.array([1,1])
    target_location =np.array([size-2, size-2])
    
    layout = np.zeros((size,size), dtype=int) * Cell.EMPTY
    layout[0,:] = Cell.WALL
    layout[-1,:] = Cell.WALL
    layout[:,0] = Cell.WALL
    layout[:,-1] = Cell.WALL
    layout[::2,::2] = Cell.WALL
    i_walls = 0
    n_walls = int(size*size/15)
    while i_walls < n_walls:
        y = np.random.randint(1,size-1)
        x = np.random.randint(0,int((size-1)/2)) * 2
        if y % 2 == 0:
            x += 1
        if layout[y,x] != Cell.WALL:
            layout[y,x] = Cell.WALL
            i_walls += 1
            
    #make sure starting and target point are empty cells
    layout[start_location[0], start_location[1]] = Cell.EMPTY
    layout[target_location[0], target_location[1]] = Cell.EMPTY
    return layout, start_location, target_location
            
def show_layout(layout_to_show):
    layout = layout_to_show
    layout[layout==0] = 255 #empty in white
    layout[layout==1] = 0 #walls in black
    plt.imshow(layout_to_show, vmin=0, vmax=255, cmap = 'gray')
    
    
'''env'''

class Cell:
    EMPTY = 0
    WALL = 1

# see https://www.gymlibrary.ml/content/environment_creation/
class Maze:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, layout, start_location, target_location, size):
        self.layout = layout
        self.layout[self.layout]
        self.size = layout.shape[0]  # The size of the square grid
        self.start = start_location
        self.target = target_location
        self.window_size = 512  # The size of the PyGame window
        
        #The observation is the agent's current coordinates in the maze
        self.observation_space = spaces.Box(0, size - 1, shape=(2,), dtype=int)
        
        
        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)
        
        #map actions to directions
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        
        #information on the experience
        self.info = None
        
        # Set the Canvas to represent the layout
        self.canvas = np.ones((self.size+2, self.size, 3)) #we let some places at the top
        for channel in range(self.canvas.shape[2]):
            self.canvas[2:, :, channel] = self.layout
        
        
    def reset(self):
        # Initialize the agent's location
        self._agent_location = self.start
        
        # Reset the Canvas 
        self.canvas = np.ones((self.size+2, self.size, 3)) #we let some places at the top
        for channel in range(self.canvas.shape[2]):
            self.canvas[2:, :, channel] = self.layout
        

        observation = self._agent_location
        return observation
    
    def get_info(self, agent_location, reward):
        return {"agent_location": agent_location,"reward": reward}
    
    def get_reward(self, next_cell, done):
        if done:
            reward=1
        elif self.layout[next_cell[0], next_cell[1]] == Cell.WALL:
            reward = -0.8
        else:
            reward = -0.04
        return reward
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        
        #next cell
        next_cell = self._agent_location + direction
        
        # An episode is done if the agent has reached the target
        done = np.array_equal(next_cell, self.target)
        
        reward = self.get_reward(next_cell, done)
        
        #update agent location - handle cases when agent hits a wall
        if self.layout[next_cell[0], next_cell[1]] != Cell.WALL: #only change location if next cell is empty
            self._agent_location = next_cell
            
        observation = self._agent_location
        
        self.info = self.get_info(self._agent_location, reward)

        return observation, reward, done, self.info
    
    
    def render(self, mode='human'):
        #initialize cv2 window
        cv2.namedWindow("Game", cv2.WINDOW_AUTOSIZE)
        
        #create a colorful image
        img = self.canvas.copy()
        img[img == Cell.EMPTY] = 255 #white
        img[img == Cell.WALL] = 0 #black
        img[:2, :, :] = 255
        
        img[self.target[0]+2, self.target[1], :] = [128, 0, 0] #put the target in blue
        img[self._agent_location[0] + 2, self._agent_location[1], :] = [0, 0, 128] #Red
        
        #upscale imaeg
        img = cv2.resize(img, (int(img.shape[1] * 10), int(img.shape[0] * 10)), interpolation=cv2.INTER_AREA)
       
        # Put the info on image
        #text = 'Location: {} | Reward: {}'.format(self.info['agent_location'], self.info['reward'])
        #img = cv2.putText(img, text, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
        
        if mode == "human":
            cv2.imshow("Game", img)
            cv2.waitKey(10)
    
        elif mode == "rgb_array":
            return img
    
    
    
