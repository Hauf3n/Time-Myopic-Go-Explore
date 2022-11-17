import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
dtype = torch.float

class Embedding_Holder():
# hold images of checkpoints and their embeddings

    def __init__(self, network, device):
            self.frames = None
            self.embeddings = None
            self.network = network
            self.device = device
            
    def add_frame(self, frame):
        
        frame = torch.tensor(frame).to(self.device).to(dtype)
        
        if self.frames == None:
            self.frames = frame
        else:
            self.frames = torch.cat((self.frames, frame),dim=0)
            
    def compute_embeddings(self):
        self.embeddings = self.network.embedding(self.frames).detach()     