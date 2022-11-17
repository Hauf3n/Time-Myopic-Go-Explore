import numpy as np
import cv2
import gym
import random
import matplotlib.pyplot as plt
import multiprocessing
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import time as t
import hashlib
import wandb

from torch.utils.data import DataLoader
from Dataset import *
from Network import *
from Archive import *
from Environment import *
from Holder import *

# globals
idx_counter = 0
device = None
dtype = torch.float
env_name = "MontezumaRevengeDeterministic-v4"    
time_model_updates = 0
                 
# multiprocessing method
def run(start_cell): 

    env_runner = Env_Runner(env_name)
    traj = env_runner.run(start_cell)
    
    # assemble data
    # collect frames and restores and score  
    frames = [start_cell.frame]
    restores = [start_cell.restore]
    scores = [start_cell.score]
    hashes = [hashlib.md5(start_cell.frame).hexdigest()]
    
    steps = len(traj)
    for i, traj_element in enumerate(traj):
    
        frame, action, reward, done, restore, hashh = traj_element
        if reward > 0:
            print("seen reward:", reward)
            
        frames.append(frame)
        restores.append(restore)
        scores.append(scores[-1] + reward)
        hashes.append(hashh)
        
    return (frames, restores, scores, hashes, steps)
    
# multiprocessing method
def run_proxies(proxy): 
    
    frame, restore = proxy 
    start_cell = Cell(-1, restore, frame)
    
    env_runner = Env_Runner(env_name)
    traj = env_runner.run(start_cell)
    
    # assemble data
    # collect frames and restores and score  
    frames = [start_cell.frame]
    restores = [start_cell.restore]
    scores = [start_cell.score]
    hashes = [hashlib.md5(start_cell.frame).hexdigest()]
    
    steps = len(traj)
    for i, traj_element in enumerate(traj):
    
        frame, action, reward, done, restore, hashh = traj_element
        if reward > 0:
            print("seen reward:", reward)
            
        frames.append(frame)
        restores.append(restore)
        scores.append(scores[-1] + reward)
        hashes.append(hashh)
        
    return (frames, restores, scores, hashes, steps)

def main(args, script_start_time, total_minutes):

    global idx_counter
    global device
    global env_name
    
    boxplot_sv = {}
    
    # create folder to save runtime information, models etc.
    #p = os.getcwd() + '/TIME_'+t.asctime(t.gmtime()).replace(" ","_").replace(":","_")+'/'
    p = os.getcwd() + f'/{wandb.run.name}'
    os.mkdir(p)
    logger = open(p+"/exploration.csv", "w")
    logger.write(f'step,score,cells\n')
    logger.close() 
    
    # INIT
    
    # wandb custom metrics
    wandb.define_metric("time_update")
    wandb.define_metric("env_steps")
    
    wandb.define_metric("time_loss", step_metric="time_update")
    wandb.define_metric("score", step_metric="env_steps")
    wandb.define_metric("archive", step_metric="env_steps")
    wandb.define_metric("archive size", step_metric="env_steps")
    wandb.define_metric("checkpoint distance ok", step_metric="env_steps")
    wandb.define_metric("candidate to close", step_metric="env_steps")
    wandb.define_metric("model training time (minutes)", step_metric="env_steps")
    
    # hyperparameter 
    time_window = args.time_window
    batch_size = args.batch_size
    epochs = args.epochs
    time_threshold = args.time_threshold
    env_name = args.env
    num_cpus = args.cpus
    device = torch.device(args.device)
    
    # init model 
    network = Time_Network(3,32,256).to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    loss =  nn.MSELoss()#nn.L1Loss()
    dataset = Dataset(time_window)
    
    # init archive
    idx_counter = 0
    env_tmp = gym.make(env_name).unwrapped
    env_tmp.seed(0)
    start_s = cv2.resize(env_tmp.reset()[30::,:].astype('float32'), dsize=(84,84))
    start_s = cv2.cvtColor(start_s, cv2.COLOR_BGR2GRAY)/255
    start_s = np.stack([start_s,start_s,start_s],axis=2)
    start_restore = env_tmp.clone_state(include_rng=True)
    start_cell_info = [idx_counter, start_restore, start_s, None]
    archive = Archive()
    archive.init_archive(start_cell_info)
    idx_counter += 1
    
    # track max score in archive
    best_score = 0 #-np.inf
    max_score_idx = 0
    
    cell_imgs = [wandb.Image(archive.cells[cell_key].frame, caption="idx: "+str(archive.cells[cell_key].idx)+" v: "+str(archive.cells[cell_key].visits)) for cell_key in archive.cells]
    wandb.log({"archive":cell_imgs, "archive size": len(archive.cells), "score":best_score,"env_steps":0})
    
    # add env start to dataset
    dataset.add_bubble(0, start_s, hashlib.md5(start_s).hexdigest(), 1000)
    
    # init selector
    selector = CellSeletor(archive)
    
    # multiprocessing
    pool = multiprocessing.Pool(num_cpus)
    
    # Inital training of the model
    # always starting point: env.reset()
    start_cells = [archive.cells[0] for i in range(100)]
    
    result = pool.map(run, start_cells)
    
    frame_trajs = []
    hash_trajs = []
    checkpoint_idxs = []
    
    for traj, start_cell in zip(result,start_cells):
        frame_traj, _, _, hash_traj, _ = traj
        frame_trajs.append(frame_traj)
        hash_trajs.append(hash_traj)
        checkpoint_idxs.append(start_cell.idx)
        
    dataset.build(checkpoint_idxs, frame_trajs, hash_trajs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=args.dataset_workers, shuffle=True, drop_last=True)
    
    optimize_model(network, optimizer, dataloader, loss, epochs)  
    print("INIT DONE")
    torch.save(network,p+f'/network_INIT.pt')
    log_model_quality(network, time_window)
    #return
    
    # build checkpoint embedding holder
    embedding_holder = Embedding_Holder(network, device)
    
    # set an embedding for start state
    archive.cells[0].embedding = network.embedding(torch.tensor(archive.cells[0].frame).to(device).to(dtype).unsqueeze(0).permute(0,3,1,2)).detach().cpu()
    embedding_holder.add_frame(torch.tensor(archive.cells[0].frame).to(dtype).unsqueeze(0).permute(0,3,1,2))
    
    steps = 0
    iteration = 0
    current_time = t.time()
    elapsed_minutes = (current_time - script_start_time)/60
    while steps < args.num_steps and elapsed_minutes < total_minutes:
        
        # wandb
        wandb_far_away = 0
        wandb_candidate_to_close = 0
        
        # get data
        start_cells = selector.select_cells(100, best_score) 
        proxies = selector.select_proxies(50, best_score) 
        
        result = pool.map(run, start_cells)
        proxy_data = pool.map(run_proxies, proxies)
        
        # collect frames and restores and score from data
        frame_trajs = []
        restore_trajs = []
        score_trajs = []
        hash_trajs = []
        checkpoint_idxs = []
        
        for traj, start_cell in zip(result,start_cells): 
            
            frame_traj, restores, scores, hash_traj, num_steps = traj
            steps += num_steps
              
            # save collected data
            frame_trajs.append(frame_traj)
            restore_trajs.append(restores)
            score_trajs.append(scores)
            hash_trajs.append(hash_traj)
            checkpoint_idxs.append(start_cell.idx)
            
        dataset.build(checkpoint_idxs, frame_trajs, hash_trajs)
        
        # collect frames and restores and score from proxies
        frame_trajss = []
        hash_trajss = []
        
        for traj in proxy_data: 
            
            frame_traj, _, _, hash_traj, num_steps = traj
            steps += num_steps
              
            # save collected data
            frame_trajss.append(frame_traj)
            hash_trajss.append(hash_traj)
        
        dataset.add_proxies(frame_trajss, hash_trajss)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=args.dataset_workers, shuffle=True, drop_last=True)
        
        # OPTIMIZE MODEL
        optimize_model(network, optimizer, dataloader, loss, epochs)
        
        # update all holder embeddings
        embedding_holder.compute_embeddings()
        
        # update only cell embeddings in archive which are selected as start cells
        seen_cells_idx = []
        seen_cells = []
        cell_frames = []
        for cell in start_cells:
            if cell.idx not in seen_cells_idx:
                seen_cells_idx.append(cell.idx)
                seen_cells.append(cell)
                cell_frames.append(cell.frame)
                
        # recompute archive embeddings
        frame_embeddings = torch.tensor(np.array(cell_frames)).to(device).to(dtype).permute(0,3,1,2)
        frame_embeddings = network.embedding(frame_embeddings).detach().cpu()
        for i, cell in enumerate(seen_cells):
            cell.embedding = frame_embeddings[i]
        
        
        # add a new checkpoint to the archive if there is progress in time        
        for frames, start_cell, restores, scores, hashes in zip(frame_trajs, start_cells, restore_trajs, score_trajs, hash_trajs):
              
            seen = []
            new_entry = []
            
            start_cell.visits += 1
            
            start_cell_score = start_cell.score
            if start_cell_score not in boxplot_sv.keys():
                boxplot_sv[start_cell_score] = 1
            else:
                boxplot_sv[start_cell_score] = boxplot_sv[start_cell_score] + 1
            
            # put element directly in archive if we see new highscore
            s = np.array(scores)
            traj_max_idx = np.argmax(s)
            traj_max_score = s[traj_max_idx]
            if traj_max_score > best_score:
                print("found new highscore. Put cell in archive!")
                # put new cell in archive
                new_parent_embedding = network.embedding(torch.tensor(frames[traj_max_idx]).to(device).to(dtype).unsqueeze(0).permute(0,3,1,2)).detach().cpu()
                add_new_cell(restores[traj_max_idx], frames[traj_max_idx], new_parent_embedding, scores[traj_max_idx], embedding_holder, archive)
                dataset.add_bubble(idx_counter-1, frames[traj_max_idx], hashes[traj_max_idx], 1000)
                
                if traj_max_score > best_score:
                    best_score = traj_max_score
                    max_score_idx = idx_counter-1
                    print("new best score: ", best_score)
                    torch.save(network,p+f'/network_{best_score}.pt')
                    
                continue
            
            # transform every seen frame into an embedding
            frame_embeddings = frames[1::]
            frame_embeddings = torch.tensor(np.array(frame_embeddings)).to(device).to(dtype).permute(0,3,1,2)
            frame_embeddings = network.embedding(frame_embeddings).detach()
            
            start_cell_embedding = start_cell.embedding.to(device)
            
            # look at time distance between start frame and all trajectory frames
            start_cell_embedding = start_cell_embedding.repeat(frame_embeddings.shape[0],1).detach()
            
            time = network.time(start_cell_embedding, frame_embeddings).detach().cpu().numpy()#.squeeze(1)
            
            # add checkpoint proxies for better data generation
            proxy_idxs = get_checkpoint_proxies(time, num_samples=2)
            for idx in proxy_idxs:
                start_cell.add_proxy(frames[idx], restores[idx])
            
            # select frames which have the greatest time distance to the start frame
            # select the best candidates in the trajectory
            best_idxs = get_traj_candidates(time, num_candidates=2, skip_front=10)
            
            # try add candidates to archive
            for best_idx in best_idxs:
                  
                # now check time distance to every embedding in the whole archive - because we dont want loops
                cell_embeddings = embedding_holder.embeddings
                
                # extract the correct frame embedding
                frame_embedding = frame_embeddings[best_idx]
                
                # compare time elapsed best_idx with all checkpoints in archive
                frame_embedding = frame_embedding.repeat(cell_embeddings.shape[0],1).detach()
                times = network.time(cell_embeddings, frame_embedding).detach().cpu().numpy()#.squeeze(1)
                
                # time to every cell in archive must be larger than time threshold
                far_away = (times > time_threshold).all()
                
                if not far_away:
                    # add a visit to checkpoints that are too close to candidate
                    checkpoints_idx_close = (times < 0.65).nonzero()[0]
                    for c_idx in checkpoints_idx_close:
                        if c_idx not in seen and c_idx not in new_entry:
                            archive.cells[c_idx].visits += 1
                            #seen.append(c_idx)
                
                # candidate is not close to other cells in archive 
                if far_away:
                    
                    wandb_far_away += 1
                    
                    # reach a threshold from candidate to every checkpoint to be accepted
                    times = network.time(frame_embedding, cell_embeddings).detach().cpu().numpy()
                    # time to every cell must be larger than time threshold
                    candidate_away = (times > 0.35).all()
                    
                    if not candidate_away:
                        wandb_candidate_to_close += 1
                        # dont accept candidate
                        continue
                
                    # put new cell in archive
                    new_parent_embedding = network.embedding(torch.tensor(frames[best_idx+1]).to(device).to(dtype).unsqueeze(0).permute(0,3,1,2)).detach().cpu()
                    add_new_cell(restores[best_idx], frames[best_idx+1], new_parent_embedding, scores[best_idx], embedding_holder, archive)
                    dataset.add_bubble(idx_counter-1, frames[best_idx+1], hashes[best_idx+1], 1000)
                    new_entry.append(idx_counter-1)
                    
                    if scores[best_idx] > best_score:
                        best_score = scores[best_idx]
                        max_score_idx = idx_counter-1
                        print("new best score: ", best_score)
                        torch.save(network,p+f'/network_{best_score}.pt')
                        
        iteration += 1 
        # reset dynamic part of the dataset after some iterations
        if iteration%7 == 0:
            #print("reset dataset")
            dataset.reset()
            
        log_model_quality(network, time_window)
            
        cell_imgs = [wandb.Image(archive.cells[cell_key].frame, caption="idx: "+str(archive.cells[cell_key].idx)+" v: "+str(archive.cells[cell_key].visits)) for cell_key in archive.cells]
        wandb.log({"archive":cell_imgs, "checkpoint distance ok":wandb_far_away,"candidate to close":wandb_candidate_to_close,"archive size": len(archive.cells),"score":best_score, "env_steps":steps})
        # track elapsed time
        current_time = t.time()
        elapsed_minutes = (current_time - script_start_time)/60 
        
    f = open(p+f'/sv_boxplot.data', 'wb')
    pickle.dump(boxplot_sv,f)
    f.close()        
    pool.terminate()

def get_checkpoint_proxies(times, num_samples=2, threshold_low=0.45, threshold_high=0.75):
    # sample traj idxs that surpass time threshold_low to start checkpoint
    
    low_idxs = (times > threshold_low).nonzero()[0]
    high_idxs = (times < threshold_high).nonzero()[0]
    
    idxs = []
    for i in low_idxs:
        if i in high_idxs:
            idxs.append(i)
    
    if len(idxs) == 0:
        return []
    
    rnd_idx = np.random.choice(idxs, num_samples)
    return rnd_idx
    

def get_traj_candidates(times, num_candidates=3, skip_front=5):
    # return idxs
    
    if len(times) < (skip_front + 3 * num_candidates) or num_candidates == 1:
        # only one idx
        max_idx = [np.argmax(times)]
        return max_idx
    
    times = np.array(times[skip_front::])
    n = len(times)
    splits = [ i*(n//num_candidates) for i in range(1, num_candidates)] 
    sub_arrays = np.split(times, splits)
    
    # relative max idx in the selected interval
    max_idxs = [np.argmax(sub_arrays[i]) for i in range(len(sub_arrays))]    
    
    # correct the idx
    max_idxs = np.add(max_idxs, skip_front)
    current_len = 0
    for i in range(1, len(sub_arrays)):
        current_len += len(sub_arrays[i-1])
        max_idxs[i] = max_idxs[i] + current_len
    
    return max_idxs
      
            
def add_new_cell(restore, frame, frame_embedding, score, embedding_holder, archive):
    global idx_counter
    # put new checkpoint in archive
                
    new_cell = Cell(idx_counter, restore, frame, frame_embedding, score=score)          
    # add new checkpoint in archive and embedding holder
    embedding_holder.add_frame(torch.tensor(frame).to(device).to(dtype).unsqueeze(0).permute(0,3,1,2))
    
    # update all holder embeddings
    embedding_holder.compute_embeddings()
    
    archive.cells[idx_counter] = new_cell 
    idx_counter += 1
    
def log_model_quality(network, time_window):
    f = open('montezuma_demo_imgs.data', 'rb')
    frames = pickle.load(f)[0:121]
    f.close()
    plt.figure()
    
    #real_imgs = [wandb.Image(frames[i]) for i in range(121)]
    #wandb.log({"original":real_imgs}, step = 9999)
    
    stacks = 3
    f = []
    frame_0 = cv2.cvtColor(frames[0].astype('float32'), cv2.COLOR_BGR2GRAY)
    observation = np.stack([frame_0,frame_0,frame_0],axis=2)
    f.append(observation.copy())
    for i in range(1,121):
        observation[:,:,0:-1] = observation[:,:,1::]
        observation[:,:,-1] = cv2.cvtColor(frames[i].astype('float32'), cv2.COLOR_BGR2GRAY)
        #f1 = cv2.cvtColor(frames[i].astype('float32'), cv2.COLOR_BGR2GRAY)
        #observation = np.stack([f1,f1,f1],axis=2)
        f.append(observation.copy())
    
    #traj_imgs = [wandb.Image(f[i]) for i in range(121)]
    #wandb.log({"traj":traj_imgs}, step = 10000)
    
    network_input = torch.tensor(np.array(f)/255).to(device).to(dtype).permute(0,3,1,2)#
    X = network.embedding(network_input).detach().cpu().numpy()
     
    idxs = [0,10,20,30,40,50]
    data_plots = []
    for idx in idxs:
        time_start = (torch.tensor(X[idx]).to(device)).repeat(X.shape[0]-(idx),1)
        time_future = torch.tensor(X[idx::]).to(device)
        times = network.time(time_start, time_future)
        times = times.detach().cpu().numpy().flatten()
        times = times * time_window
        
        plt.plot(np.arange(idx,X.shape[0],1), times, linewidth=0.9,label=f'Start: {idx}')
    plt.ylim(0)    
    wandb.log({"clean trajectory prediction":wandb.Image(plt)})
    
    
    #table = wandb.Table(data=data_plots, columns=["timestep","elapsed time"])
    #wandb.log({"my_custom_plot_id" : wandb.plot.line(table, "timestep", "elapsed time", title="test")})
    

def optimize_model(network, optimizer, dataloader, loss_objective, epochs):
    global time_model_updates
    
    model_training_start_time = t.time()
    
    # set network to train
    network.train()
    
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            p, c, time_label = batch
            p, c, time_label = p.to(device).to(dtype), c.to(device).to(dtype), time_label.to(device).float()
            p, c = p.permute(0,3,1,2), c.permute(0,3,1,2)

            time = network(p, c)

            loss = loss_objective(time, time_label.unsqueeze(1))
            #print("loss: ", loss)
            wandb.log({"loss": loss})
            loss.backward()
            optimizer.step()
            
    # set network to eval      
    network.eval() 
    
    model_training_end_time = t.time()
    elapsed_time_seconds = model_training_end_time - model_training_start_time
    wandb.log({"model training time (minutes)":elapsed_time_seconds/60})
    
    
if __name__ == '__main__':

    args = argparse.ArgumentParser()
    
    # set hyperparameter
    args.add_argument('-lr', type=float, default=1e-4)
    args.add_argument('-env', default='MontezumaRevengeDeterministic-v4')
    args.add_argument('-device', default='cuda:0')
    args.add_argument('-batch_size', type=int, default=64)
    args.add_argument('-time_window', type=int, default=20)
    args.add_argument('-hours', type=int, default=48)
    args.add_argument('-minutes', type=int, default=00)
    args.add_argument('-time_threshold', type=float, default=0.6)
    args.add_argument('-num_steps', type=int, default=3000000)
    args.add_argument('-runs', type=int, default=20)
    args.add_argument('-cpus', type=int, default=16)
    args.add_argument('-epochs', type=int, default=3)
    args.add_argument('-dataset_workers', type=int, default=16)
    
    args = args.parse_args()
    
    # calculate time when we terminate script
    total_minutes = int(args.hours*60) + args.minutes
    script_start_time = t.time()
    
    print("Script will run ", args.hours, "hours and ", args.minutes," minutes. (",total_minutes," minutes)")
    print("Scheduled runs:", args.runs)
    
    for i in range(args.runs):
        #wandb.config.update(args)
        current_time = t.time()
        elapsed_minutes = (current_time - script_start_time)/60
        if elapsed_minutes < total_minutes:
            wandb_run = wandb.init(project="project-name", entity="my_wandb_entity", reinit=True)
            main(args, script_start_time, total_minutes)#,end_time)
            wandb_run.finish()

