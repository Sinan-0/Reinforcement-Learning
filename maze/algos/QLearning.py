import numpy as np

EPSILON = 0.01
ALPHA = 0.5
GAMMA = 1.0

nb_eps = 100
def run_q(env, nb_eps, EPSILON, ALPHA, GAMMA, verbose=True):
    def tuple_to_int(SIZE, _tuple):
        """Convert coordinqates into unique key --> a state will be an integer that represents a coordinate
        see https://math.stackexchange.com/questions/1588601/create-unique-identifier-from-close-coordinates
        """
        return _tuple[1] + _tuple[0]*SIZE

    def argmax(q_values):
            """argmax with random tie-breaking
            Args:
                q_values (Numpy array): the array of action-values
            Returns:
                action (int): an action with the highest value
            """
            top = float("-inf")
            ties = []

            for i in range(len(q_values)):
                if q_values[i] > top:
                    top = q_values[i]
                    ties = []

                if q_values[i] == top:
                    ties.append(i)

            return int(np.random.choice(ties))

    def init_q(SIZE, n_actions=4):
        """
        Initialize the q vector
        """
        q = np.zeros((SIZE*SIZE, n_actions)) #n_states = SIZE*SIZE
        return q

    def select_eps_greedy_action(current_q, epsilon, n_actions=4):
        """
        epsilon-greedy action
        """
        if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
        else:
            action = argmax(current_q)
        return action

    def update_q(q, SIZE, obs, action, reward, new_obs, done, alpha, gamma):
        state_q = tuple_to_int(SIZE, obs)
        new_state_q = tuple_to_int(SIZE, new_obs)
        if done: #don't look at the next state
            q[state_q, action] += alpha*(reward - q[state_q, action]) 
        else:
            q[state_q, action] += alpha*(reward + gamma*q[new_state_q, argmax(q[new_state_q, :])] - q[state_q, action]) 
        return q

    q = init_q(env.size) #initialize q
    nb_steps_to_finish = []
    all_state_visits = [] #contains number of visits per state for last 10 episodes
    for episode in range(nb_eps):
        done=False
        obs = env.reset() #reset environment
        state = tuple_to_int(env.size, obs)  #convert coordinate into integer
        state_visits = np.zeros(env.size*env.size) #contains number of visits per state
        state_visits[state] +=1 #add the visit for the starting point
        nb_step = 0
        while not done:
            current_q = q[state, :] #compute q_values in that state
            action = select_eps_greedy_action(current_q, EPSILON)
            new_obs, new_reward, done, info = env.step(action)
            q = update_q(q, env.size, obs, action, new_reward, new_obs, done, ALPHA, GAMMA)
            
            
            #show evolution
            env.render(mode='human')
            
            # update for next iteration
            state = tuple_to_int(env.size, new_obs) #update state
            obs = new_obs
            nb_step+=1
            
            #qdd the visits per state for last 10 episodes
            if episode+1 >= nb_eps-10:
                state_visits[state] +=1
            
            #time.sleep(0.25)
            
        #add result info
        if episode >= nb_eps-10:
            all_state_visits.append(state_visits)
        nb_steps_to_finish.append(nb_step)
        
        if verbose:
            print('Episode {}  | Nb Steps to finish: {}'.format(episode + 1, nb_step))
    
    return q, nb_steps_to_finish, all_state_visits



def run_double_q(env, nb_eps, EPSILON, ALPHA, GAMMA, verbose=True):
    def tuple_to_int(SIZE, _tuple):
        """Convert coordinqates into unique key --> a state will be an integer that represents a coordinate
        see https://math.stackexchange.com/questions/1588601/create-unique-identifier-from-close-coordinates
        """
        return _tuple[1] + _tuple[0]*SIZE

    def argmax(q_values):
            """argmax with random tie-breaking
            Args:
                q_values (Numpy array): the array of action-values
            Returns:
                action (int): an action with the highest value
            """
            top = float("-inf")
            ties = []

            for i in range(len(q_values)):
                if q_values[i] > top:
                    top = q_values[i]
                    ties = []

                if q_values[i] == top:
                    ties.append(i)

            return int(np.random.choice(ties))

    def init_q(SIZE, n_actions=4):
        """
        Initialize the q vector
        """
        q = np.zeros((SIZE*SIZE, n_actions))
        return q

    def select_eps_greedy_action(current_q, epsilon, n_actions=4):
        """
        epsilon-greedy action
        """
        if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
        else:
            action = argmax(current_q)
        return action

    def update_q(q1, q2, SIZE, obs, action, reward, new_obs, done, alpha, gamma):
        state = tuple_to_int(SIZE, obs)
        new_state = tuple_to_int(SIZE, new_obs)
        if np.random.random() < 0.5:
            q1[state, action] += alpha*(reward + (1-done)*gamma*q2[new_state, argmax(q1[new_state, :])] - q1[state, action]) 
        else:
            q2[state, action] += alpha*(reward + (1-done)*gamma*q1[new_state, argmax(q2[new_state, :])] - q2[state, action])
            
        return q1, q2

    q1 = init_q(env.size) #initialize q1
    q2 = init_q(env.size) #initialize q2
    nb_steps_to_finish = []
    all_state_visits = [] #contains number of visits per state for last 10 episodes
    for episode in range(nb_eps):
        done=False
        obs = env.reset() #reset environment
        state = tuple_to_int(env.size, obs)  #convert coordinate into integer
        state_visits = np.zeros(env.size*env.size) #contains number of visits per state
        state_visits[state] +=1 #add the visit for the starting point
        nb_step = 0
        while not done:
            current_q = q1[state, :] + q2[state, :] #compute q_values in that state
            action = select_eps_greedy_action(current_q, EPSILON)
            new_obs, new_reward, done, info = env.step(action)
            q1, q2 = update_q(q1, q2, env.size, obs, action, new_reward, new_obs, done, ALPHA, GAMMA)
            
            
            #show evolution
            env.render(mode='human')
            
            # update for next iteration
            state = tuple_to_int(env.size, new_obs) #update state
            obs = new_obs
            nb_step+=1
            
            #qdd the visits per state for last 10 episodes
            if episode+1 >= nb_eps-10:
                state_visits[state] +=1
            
            #time.sleep(0.25)
            
        #add result info
        if episode >= nb_eps-10:
            all_state_visits.append(state_visits)
        nb_steps_to_finish.append(nb_step)
        
        if verbose:
            print('Episode {}  | Nb Steps to finish: {}'.format(episode + 1, nb_step))
    
    return q1, q2, nb_steps_to_finish, all_state_visits