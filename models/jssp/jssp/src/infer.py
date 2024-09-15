import torch
from torch import optim
import numpy as np
from .rl.Models.actorcritic import *


def load_model(model_path,
               jobs,
               ops):
    device="cpu"
    # input_model = './output_models/' + model_path + '.tar'
    input_model = model_path
    embeddim = 128
    jobs = int(jobs)
    ops = int(ops)
    macs = int(ops)

    # Instanitate actor-critic
    actor = Actor(embeddim,jobs,ops,macs,device).to(device)
    critic = Critic(embeddim,jobs,ops,macs,device).to(device)

    # Environment training

    actor_opt = optim.Adam(actor.parameters())
    critic_opt = optim.Adam(critic.parameters())

    checkpoint = torch.load(input_model, map_location=torch.device('cpu'))
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
    critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])

    model = {"actor":actor,"critic":critic}
    return model

def infer(model,jobs,ops,problemargs,stateargs,):
  
    jobs = int(jobs)
    ops = int(ops)

    stateargs = dict(stateargs)
    problemargs = dict(problemargs)
    # print(stateargs,problemargs,"\n")

    # rlstate = {
    #         'machine_utilization': np.expand_dims(np.array([2 for i in range(ops)])/100, axis=0),
    #         'job_times': np.expand_dims(np.array([[1 for i in range(ops)] for j in range(jobs)])/100, axis=0),
    #         'job_early_start_time': np.expand_dims(np.array([2 for i in range(jobs)])/100, axis=0),
    #         'precedence': np.expand_dims(np.array([[1 for i in range(ops)] for j in range(jobs)]), axis=0),
    #         'job_state': np.expand_dims(np.array([1 for i in range(jobs)]), axis=0)
    #     }

    # stateargs = {'machine_utilization': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'job_early_start_time': [0, 0], 'job_state': [28, 28]} 
    # problemargs = {'job_times': [[160, 5, 139, 99, 9, 98, 28, 107, 196, 165, 114, 7, 34, 133, 76], [105, 160, 19, 189, 25, 95, 15, 122, 165, 2, 66, 111, 51, 83, 183]], 'precedence': [[0, 13, 6, 11, 12, 5, 2, 1, 3, 10, 7, 4, 14, 8, 9], [14, 7, 3, 2, 11, 1, 12, 0, 4, 9, 10, 13, 8, 6, 5]]} 


    # stateargs = {'machine_utilization': [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], 'job_early_start_time': [0, 2], 'job_state': [28, 29]} 
    # problemargs = {'job_times': [[160, 5, 139, 99, 9, 98, 28, 107, 196, 165, 114, 7, 34, 133, 76], [105, 160, 19, 189, 25, 95, 15, 122, 165, 2, 66, 111, 51, 83, 183]], 'precedence': [[0, 13, 6, 11, 12, 5, 2, 1, 3, 10, 7, 4, 14, 8, 9], [14, 7, 3, 2, 11, 1, 12, 0, 4, 9, 10, 13, 8, 6, 5]]} 

    rlstate = {
            'machine_utilization': np.expand_dims(np.array(stateargs['machine_utilization'])/100, axis=0),
            'job_times': np.expand_dims(np.array(problemargs['job_times'])/100, axis=0),
            'job_early_start_time': np.expand_dims(np.array(stateargs['job_early_start_time'])/100, axis=0),
            'precedence': np.expand_dims(np.array(problemargs['precedence']), axis=0),
            'job_state': np.expand_dims(np.array(stateargs['job_state']), axis=0)
        }
    # apply mask
    invalid_mask = (np.array(stateargs['job_state']) < ops).astype(int)

    # Compute action
    with torch.set_grad_enabled(False): # False as not training
        actorJobEmb = model["actor"].instance_embed(rlstate)
        criticJobEmb = model["critic"].instance_embed(rlstate)
        prob, log_prob = model["actor"](rlstate,actorJobEmb,1,invalid_mask)
        value = model["critic"](rlstate,criticJobEmb)
    probs_product = prob.cpu() 
    argmax_prob = np.dstack(np.unravel_index(np.argsort(probs_product.ravel()), probs_product.shape))[0][::-1]

    # call forward to get action
    chosenJob = [argmax_prob[0, 1]][0]

    return chosenJob