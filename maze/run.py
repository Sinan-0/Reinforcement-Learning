#algos
from algos.QLearning import run_q, run_double_q
from algos.SARSA import run_ES
from algos.DDQN import *
from algos.NStep_A2C import *
from algos.OneStep_A2C import *

#environment
from env import *

#rendering the environment
import cv2

#argparse
import argparse

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--size", required=True, type=int, 
   help="length of the square maze (if size=10, the maze will be a 10*10 square)")
ap.add_argument("--algo", required=True, type=str,
   help="Algorithm to use : QLearning, DoubleQLearning, ExpectedSARSA, DDQN, NStepA2C, 1StepA2C")

ap.add_argument("--nb_eps", required=False, type=int, default=100,
   help="Number of episodes")
ap.add_argument("--epsilon", required=False, type=float, default=0.01,
   help="Epsilon value")
ap.add_argument("--alpha", required=False, type=float, default=0.5,
   help="Alpha value")
ap.add_argument("--gamma", required=False, type=float, default=1.0,
   help="Gamma value")
ap.add_argument("--max_steps", required=False, type=int, default=1500,
   help="max_steps an agent performs before re-starting the environment")
ap.add_argument("--batch_size", required=False, type=int, default=32)
args = vars(ap.parse_args())

nb_eps = args['nb_eps']
EPSILON = args['epsilon']
ALPHA = args['alpha']
GAMMA = args['gamma']

max_steps = args['max_steps']
batch_size = args['batch_size']


def create_env(SIZE):
    '''
    Create environment
    Args:
        -SIZE: size of the maze
    '''
    layout, start, target = create_maze(SIZE)
    env = Maze(layout, start, target, size=SIZE)
    return env



if __name__ == "__main__":
    #create environment
    env = create_env(args['size'])
    
    #run algorithm
    if args['algo'] == "QLearning":
        print('yess')
        q, nb_steps_to_finish, all_state_visits = run_q(env, nb_eps, EPSILON, ALPHA, GAMMA, verbose=False)
        
    if args['algo'] == "DoubleQLearning":
        q1, q2, nb_steps_to_finish, all_state_visits = run_double_q(env, nb_eps, EPSILON, ALPHA, GAMMA, verbose=False)
        
    if args['algo'] == "ExpectedSARSA":
        q, nb_steps_to_finish, all_state_visits = run_ES(env, nb_eps, EPSILON, ALPHA, GAMMA, verbose=False)
        
    if args['algo'] == "DDQN":
        agent = DQNAgent(env)
        nb_steps_to_finish, all_state_visits = run_DDQN(env, agent, nb_eps, max_steps, batch_size, verbose = False)
        
    if args['algo'] == "NStepA2C":
        agent = N_Step_A2CAgent(env, learning_rate=1e-3, gamma=1)
        agent, nb_steps_to_finish, all_state_visits = run_nstep_a2c(env, agent, nb_eps, max_steps, verbose = False)
        
    if args['algo'] == "1StepA2C":
        agent = OneStep_A2CAgent(env, learning_rate = 1e-5)
        agent, nb_steps_to_finish, all_state_visits = run_a2c(env, agent, nb_eps, max_steps, verbose = False)
        
    cv2.destroyAllWindows()