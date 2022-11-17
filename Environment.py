import numpy as np
import hashlib
import gym
import cv2

class Env_Actor():
    # suggested paper actor - random action repeating actor
    # sample from bernoulli distribution with p = 1/mean for action repetition
    
    def __init__(self, env, mean_repeat=4):#7
        self.num_actions = env.action_space.n
        self.mean_repeat = mean_repeat
        self.env = env
        
        self.current_action = self.env.action_space.sample()
        self.repeat_action = np.random.geometric(1 / self.mean_repeat)
        
    def get_action(self):
        
        if self.repeat_action > 0:
            self.repeat_action -= 1
            return self.current_action
            
        self.current_action = self.env.action_space.sample()
        self.repeat_action = np.random.geometric(1 / self.mean_repeat) - 1
        return self.current_action
    
class Env_Runner():
    # agent env loop
    
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name).unwrapped
        self.actor = Env_Actor(self.env)
        self.env.seed(0)
        
    def run(self, start_cell, max_steps=40):
        
        self.env.restore_state(start_cell.restore)
        
        # shape h,w,c*k
        observation = start_cell.frame
         
        traj_elemtents = []
        step = 0
        done = False
        while not done and step < max_steps:
            
            # collect data
            action = self.actor.get_action()
            frame, reward, d, _ = self.env.step(action)
            # resize frame
            frame = cv2.resize(frame[30::,:].astype('float32'), dsize=(84,84))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255
            
            # shift input in observation
            observation[:,:,0:-1] = observation[:,:,1::]
            observation[:,:,-1] = frame
            
            #observation = np.stack([frame,frame,frame],axis=2)
            #print(observation.shape)
            
            #obs_out = observation
            #obs_out[:,:,0] = obs_out[:,:,0] - obs_out[:,:,1]
            #obs_out[:,:,1] = obs_out[:,:,1] - obs_out[:,:,2]
            
            restore = self.env.clone_state(include_rng=True)
            
            # save data
            #traj_element = (obs_out, action, reward, d, restore, hashlib.md5(obs_out).hexdigest())
            traj_element = (observation.copy(), action, reward, d, restore, hashlib.md5(observation.copy()).hexdigest())
            traj_elemtents.append(traj_element)
            
            if d:
                done = True
            step += 1
            
        return traj_elemtents         