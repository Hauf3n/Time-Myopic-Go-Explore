import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Cell():
    # item in archive
    
    def __init__(self, idx, restore, frame, embedding=None, score=-np.inf, proxies=14):
    
        self.visits = 0
        self.idx = idx
        self.restore = restore
        self.embedding = embedding
        self.score = score
        self.frame = frame
        
        self.checkpoint_proxies = []
        self.i_p = 0
        self.num_proxies = proxies
        
    def add_proxy(self, frame, restore):
        
        if len(self.checkpoint_proxies) <= self.num_proxies:
            self.checkpoint_proxies.append((frame, restore))
            self.i_p += 1
        else:
            if self.i_p >= self.num_proxies:
                self.i_p = 0
                
            self.checkpoint_proxies[self.i_p] = (frame, restore)
            self.i_p += 1
        
class Archive():
    def __init__(self):
        # idx | cell
        self.cells = {}
        
    def __iter__(self):
        return iter(self.cells)
    
    def init_archive(self, start_info):
        self.cells = {}
        # start cell
        self.cells[start_info[0]] = Cell(start_info[0],start_info[1],start_info[2],
                                         start_info[3], score=0)

class CellSeletor():
    # select starting cells
    
    def __init__(self, archive):
        self.archive = archive
        
    def select_cells(self, amount, best_score):
        keys = []
        weights = []
        for key in self.archive.cells:
            if key == None: # done cell
                weights.append(0.0)
            else:
                w_visits = 1/(np.sqrt(self.archive.cells[key].visits)+1)
                w_score = np.maximum(self.archive.cells[key].score / (best_score+1), 0.075)
                weights.append(w_visits * w_score)
            keys.append(key)
            
        indexes = np.random.choice(range(len(weights)),size=amount,p=weights/np.sum(weights))
        
        selected_cells = []
        for i in indexes:
            selected_cells.append(self.archive.cells[keys[i]])
        return selected_cells

    def select_proxies(self, amount, best_score):
        # select checkpoint from which to sample proxies
        # sample based on checkpoint visits
        
        weights = []
        keys = []
        for key in self.archive.cells:
            w_visits = 1/(np.sqrt(self.archive.cells[key].visits)+1)
            w_score = np.maximum(self.archive.cells[key].score / (best_score+1), 0.075)
            weights.append(w_visits * w_score)
            #weights.append(1/(np.sqrt(self.archive.cells[key].visits)+1))
            keys.append(key)
            
        cell_keys = np.random.choice(range(len(weights)),size=amount,p=weights/np.sum(weights))
        
        proxies = []
        for key in cell_keys:
            cell = self.archive.cells[key]
            cell_proxies = cell.checkpoint_proxies
            num_proxies = len(cell_proxies)
            
            if num_proxies == 0:
                proxies.append((cell.frame, cell.restore))
            else:
                proxy_idx = np.random.choice(range(num_proxies), size=1)[0]
                proxies.append(cell.checkpoint_proxies[proxy_idx])
            
        return proxies