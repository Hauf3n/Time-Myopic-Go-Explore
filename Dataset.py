import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Checkpoint_Bubble():

    def __init__(self, size, checkpoint_obs, checkpoint_md5):
        super().__init__()
        
        self.checkpoint_obs = checkpoint_obs   
        self.checkpoint_md5 = checkpoint_md5
        self.obs = []
        self.md5s = []
        self.labels = []
        self.i = 0
        self.size = size
        
    def add(self, ob, md5, label):
        
        
        if md5 in self.md5s:
            index = self.md5s.index(md5)
            if label < self.labels[index]:
                self.obs[index] = ob
                self.md5s[index] = md5
                self.labels[index] = label
            return
            
        if np.random.rand(1)[0] > 0.33:
            return
            
        if len(self.obs) <= self.size:
            self.obs.append(ob)
            self.md5s.append(md5)
            self.labels.append(label)
            self.i += 1
        else:
            if md5 not in self.md5s:
                if self.i >= self.size:
                    self.i = 0
                    
                self.obs[self.i] = ob
                self.md5s[self.i] = md5
                self.labels[self.i] = label
                self.i += 1
                              
    def __len__(self):
        return len(self.obs)
    
    def get_checkpoint_obs(self):
        return self.checkpoint_obs
    
    def get(self, i):
        return self.obs[i], self.labels[i]
        
    def get_pair(self, i):
        return self.checkpoint_obs, self.obs[i], self.labels[i]
        
class Dataset(torch.utils.data.Dataset):

    def __init__(self, time_window, bubble_size=400):
        super().__init__()  

        # key: checkpoint idx, bubble class
        self.bubbles = {}
        self.bubble_size = bubble_size
        self.time_window = time_window
        self.length = 0
        self.reset()
        
    def __len__(self):
        return self.length + int(len(self.bubbles) * np.sqrt(self.bubble_size))
              
    def add_bubble(self, idx, checkpoint_obs, checkpoint_md5, size):
        if idx not in self.bubbles.keys():
            self.bubbles[idx] = Checkpoint_Bubble(size, checkpoint_obs, checkpoint_md5)
    
    def bubble_add_entry(self, idx, obs, md5, label):
        self.bubbles[idx].add(obs, md5, label)
    
    def reset(self):
        # key: md5 hash | value: dict -> key: md5 hash | value: time integer
        self.window_labels = {}
        self.imgs = {}
        self.imgs_c_md5 = {}
        self.md5s = []
    
    def build(self, checkpoint_idxs, trajs, hashes):
        # build temporary dataset and extend or update the bubbles
        
        # calculate window labels
        self.trajs = trajs
        self.hash_trajs = hashes
        
        # collect md5s, imgs
        for i in range(len(self.hash_trajs)):
            cur_traj = self.hash_trajs[i]
            for j in range(len(cur_traj)):
            
                md5 = cur_traj[j]
                if md5 not in self.md5s:
                    self.md5s.append(md5)
                if md5 not in self.imgs.keys():
                    self.imgs[md5] = self.trajs[i][j]
                    self.imgs_c_md5[md5] = self.hash_trajs[i][0]
        
        # fill the windows, bubbles for each observation
        for i in range(len(self.hash_trajs)):
            cur_traj = self.hash_trajs[i]
            
            # add a bubble if checkpoint has none
            #self.add_bubble(checkpoint_idxs[i], self.trajs[i][0], self.hash_trajs[i][0], self.bubble_size)
            
            for time_idx in range(len(cur_traj)-1):
                
                parent_md5 = cur_traj[time_idx]
                
                # add elements to bubble
                if time_idx > 1 and time_idx < self.time_window:
                    self.bubble_add_entry(checkpoint_idxs[i],
                                            self.trajs[i][time_idx],
                                            parent_md5,time_idx)
                
                # add parent md5 to dict
                if parent_md5 not in self.window_labels.keys():
                    self.window_labels[parent_md5] = {}
                
                window_md5s = None
                if time_idx + self.time_window < len(cur_traj):
                    window_md5s = cur_traj[time_idx:time_idx+self.time_window]
                else:
                    window_md5s = cur_traj[time_idx::]
                
                # add childs
                for time_distance, child_md5 in enumerate(window_md5s):
                    
                    time_label = time_distance
                    
                    if child_md5 == parent_md5:
                        continue
                    
                    # add md5 child to md5 parent
                    if child_md5 not in self.window_labels[parent_md5].keys():
                        self.window_labels[parent_md5][child_md5] = time_label
                    else: # take minimum time label 
                        self.window_labels[parent_md5][child_md5] = np.minimum(time_label,self.window_labels[parent_md5][child_md5])
                
        # calc length
        self.parents = list(self.window_labels.keys())
        self.length = len(self.parents)
        
        #print(self.bubbles.keys())
        
    def __getitem__(self, idx):
        
        # 3 branches
        # 1. bubble pair | 2. temporary dataset pair | 3. random checkpoint -> temporary dataset element  
        p_frame, c_frame, label = None, None, None
        
        pair_idx = np.random.choice([1,2,3], 1, p=[0.45,0.45,0.1])[0]
        #pair_idx = np.random.choice([1,2,3], 1, p=[1,0,0])[0]
        
        if pair_idx == 1:
            p_frame, c_frame, label = self.get_bubble_pair()
        elif pair_idx == 2:
            p_frame, c_frame, label = self.get_temp_pair()
        else:
            p_frame, c_frame, label = self.get_check_temp_pair()
            
        return self.img_augmentation(p_frame), self.img_augmentation(c_frame), label   
        
    def get_bubble_pair(self):       
        #random bubble and random element in bubble
        bubble_idx = np.random.choice(np.arange(len(self.bubbles)),1)[0]
        
        # skip when bubble is empty
        if len(self.bubbles[bubble_idx].obs) < 1:
            return self.get_bubble_pair()
            
        bubble_element_idx = np.random.choice(np.arange(len(self.bubbles[bubble_idx].obs)), 1)[0]
        
        return self.bubbles[bubble_idx].checkpoint_obs,self.bubbles[bubble_idx].obs[bubble_element_idx],torch.tensor(self.bubbles[bubble_idx].labels[bubble_element_idx])/ self.time_window 
            
    def get_temp_pair(self):
        
        # get parent md5
        idx = np.random.choice(np.arange(len(self.parents)),1)[0]
        parent_md5 = self.parents[idx]
        
        # time window or random?
        select_from_window = np.random.rand(1)[0]
        if select_from_window < 0.55:
        # take from window
        
            # take a child from parent
            child_md5s = list(self.window_labels[parent_md5].keys())
            
            if len(child_md5s) == 0:
                return self.__getitem__(1)
            
            # select child
            idx = np.random.choice(len(child_md5s),1)[0]
            child_md5 = child_md5s[idx]
            
            # get label
            label = self.window_labels[parent_md5][child_md5] / self.time_window
            
            # get frames
            p_frame = self.imgs[parent_md5]
            c_frame = self.imgs[child_md5]
            
            return p_frame, c_frame, torch.tensor(label)
        else:
        # take random
        
            # take random md5
            idx = np.random.choice(len(self.md5s),1)[0]
            md5 = self.md5s[idx]
            
            # md5 not allowed to be in parent window
            while md5 in self.window_labels[parent_md5].keys():
                # take random md5
                idx = np.random.choice(len(self.md5s),1)[0]
                md5 = self.md5s[idx]
            
            # get label            
            if md5 == parent_md5:
                label = 0.0 # zero time distance
            else:
                label = 1.0 # max time distance
                
            # get frames
            p_frame = self.imgs[parent_md5]
            c_frame = self.imgs[md5]

            # identity or stay random?
            select_identity = np.random.rand(1)[0]
            if select_identity < 0.4:
                c_frame = p_frame
                label = 0.0
        
            return p_frame, c_frame, torch.tensor(label) 
        
        
    def get_check_temp_pair(self):
        # get random checkpoint
        bubble_idx = np.random.choice(np.arange(len(self.bubbles)),1)[0]
        checkpoint_obs = self.bubbles[bubble_idx].checkpoint_obs
        # get random temporary image
        #print("HERE:",np.array(self.imgs.keys()).shape)
        obs_key = np.random.choice(list(self.imgs.keys()),1)[0]
        obs = self.imgs[obs_key]
        
        if self.imgs_c_md5[obs_key] == self.bubbles[bubble_idx].checkpoint_md5:
            return self.get_temp_pair()#self.__getitem__(1)
            
        # label
        label = 1.0
        
        return checkpoint_obs, obs, torch.tensor(label)

    def img_augmentation(self, img, x_max=4, y_max=4):
        #return img
        # img shape: (84,84,3)
        y,x,c = np.array(img).shape
        
        # shift image a few pixels
        x_shift = np.random.choice(np.arange(x_max+1),1)[0]
        y_shift = np.random.choice(np.arange(y_max+1),1)[0]
        
        # for now only at left and bottom (key on top)
        img_cutout = img[0:-(y_shift)-1,0:-(x_shift)-1]
        
        # insert shifting not always at the top corner
        x_pos = np.random.choice(np.arange(x_shift+1),1)[0]
        y_pos = np.random.choice(np.arange(y_shift+1),1)[0]
        
        img_aug = np.zeros_like(img)
        img_aug[y_pos:y-y_shift-1+y_pos,x_pos:x-x_shift-1+x_pos] = img_cutout

        return img_aug
    
    def add_proxies(self, trajs, hashes):
        # extend temporary dataset with proxies traj data

        self.trajs = trajs
        self.hash_trajs = hashes
        
        # collect md5s, imgs
        for i in range(len(self.hash_trajs)):
            cur_traj = self.hash_trajs[i]
            for j in range(len(cur_traj)):
            
                md5 = cur_traj[j]
                if md5 not in self.md5s:
                    self.md5s.append(md5)
                if md5 not in self.imgs.keys():
                    self.imgs[md5] = self.trajs[i][j]
                    self.imgs_c_md5[md5] = self.hash_trajs[i][0]
        
        # fill the windows, bubbles for each observation
        for i in range(len(self.hash_trajs)):
            cur_traj = self.hash_trajs[i]
             
            for time_idx in range(len(cur_traj)-1):
                
                parent_md5 = cur_traj[time_idx]  
                
                # add parent md5 to dict
                if parent_md5 not in self.window_labels.keys():
                    self.window_labels[parent_md5] = {}
                
                window_md5s = None
                if time_idx + self.time_window < len(cur_traj):
                    window_md5s = cur_traj[time_idx:time_idx+self.time_window]
                else:
                    window_md5s = cur_traj[time_idx::]
                
                # add childs
                for time_distance, child_md5 in enumerate(window_md5s):
                    
                    time_label = time_distance
                    
                    if child_md5 == parent_md5:
                        continue
                    
                    # add md5 child to md5 parent
                    if child_md5 not in self.window_labels[parent_md5].keys():
                        self.window_labels[parent_md5][child_md5] = time_label
                    else: # take minimum time label    
                        self.window_labels[parent_md5][child_md5] = np.minimum(time_label,self.window_labels[parent_md5][child_md5])
                
        # calc length
        self.parents = list(self.window_labels.keys())
        self.length = len(self.parents)    