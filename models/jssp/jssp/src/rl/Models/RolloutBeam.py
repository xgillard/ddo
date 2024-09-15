from src.rl.Models.StateLSTM import StateLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
from copy import deepcopy

class RolloutBeam:
    def __init__(self,venv,actor,critic,device,masking):
        
        self.venv = venv
        self.actor = actor
        self.critic = critic
        self.device = device
        self.masking = masking
        
    def play(self,BS,BSind,training=True,size_search=-1):
        
        total_reward = 0.0
        States = []
        Log_Prob = []
        Prob = []
        Action = []
        Value = []
        tr_reward = []
        entropies = []
        makespans = []
        
        ite = 0
        probs_product = torch.ones(1, self.venv.maxJobs)

        while not self.venv.multidone:
        
            mac_utl = np.array([i['machine_utilization'] for i in self.venv.BState])
            job_time = np.array([i['job_times'] for i in self.venv.BState])                 # Duration matrix
            job_early = np.array([i['job_early_start_time'] for i in self.venv.BState])
            job_state = np.array(self.venv.BSjobstate)
            pre = np.array([i['precedence'] for i in self.venv.BState])                     # Machine matrix  

            State = {
                'machine_utilization': mac_utl,
                'job_times': job_time,
                'job_early_start_time': job_early,
                'precedence': pre,
                'job_state': job_state
            }

            States.append(State)
 
            # Compute action
            with torch.set_grad_enabled(training):
                actorJobEmb = self.actor.instance_embed(State)
                criticJobEmb = self.critic.instance_embed(State)

                prob, log_prob = self.actor(State,actorJobEmb,self.masking)
                value = self.critic(State,criticJobEmb)

            m = Categorical(prob)

            entropy = m.entropy().cpu().detach().numpy()
            BSind = [i for i in range(size_search)]

            # argmax from probs_product
            # probs_product[0].shape = size_seach
            # probs_product[1].shape = num actions

            probs_product = probs_product * prob.cpu() 
            argmax_prob = np.dstack(np.unravel_index(np.argsort(probs_product.ravel()), probs_product.shape))[0][::-1]
            ID = []
            action = []
            envs = []

            new_probs_product = torch.ones(size_search, prob.shape[1])

            for k in range(size_search):
                ID.append([argmax_prob[k, 0]])
                action.append([argmax_prob[k, 1]])
                new_env = deepcopy(self.venv.envs[argmax_prob[k, 0]])
                envs.append(new_env)
                new_probs_product[k] *= probs_product[argmax_prob[k, 0]][argmax_prob[k, 1]]
            probs_product = new_probs_product/new_probs_product.sum()

            self.venv.envs = envs
            
            log_prob = log_prob[ID,action]
            prob = prob[ID,action]

            # List append
            Action.append([i[0] for i in action])
            Prob.append(prob)
            Log_Prob.append(log_prob.squeeze(1))
            Value.append(value.squeeze(1))
            entropies.append(entropy)

            # Environment step
            self.venv.faststep(BSind,[i[0] for i in action])

            # Collect reward
            tr_reward.append(self.venv.BSreward)
            total_reward+=np.array(self.venv.BSreward)

            ite+=1
        
        return ite,total_reward,States,Log_Prob,Prob,Action,Value,tr_reward, np.mean(entropies)