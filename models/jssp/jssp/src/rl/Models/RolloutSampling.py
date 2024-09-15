from src.rl.Models.StateLSTM import StateLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
from copy import deepcopy

class RolloutSampling:
    def __init__(self,venv,actor,critic,device,masking):
        
        self.venv = venv
        self.actor = actor
        self.critic = critic
        self.device = device
        self.masking = masking
        
    def play(self,BS,BSind,training=True,size_search=1):
        
        total_reward = 0.0
        States = []
        Log_Prob = []
        Prob = []
        Action = []
        Value = []
        tr_reward = []
        entropies = []
        
        ite = 0
         
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
                if ite<=1: 
                    actorJobEmb = self.actor.instance_embed(State)
                    criticJobEmb = self.critic.instance_embed(State)

                prob, log_prob = self.actor(State,actorJobEmb,self.masking)
                value = self.critic(State,criticJobEmb)

            m = Categorical(prob)

            action = m.sample().unsqueeze(1).cpu().numpy().tolist()
            entropy = m.entropy().cpu().detach().numpy()
            BSind = [i for i in range(size_search)]
            ID = [[i] for i in range(size_search)]

            if ite==0:

                action = []
                envs = []

                for k in range(size_search):
                    a = m.sample().unsqueeze(1).cpu().numpy().tolist()
                    action.append(a[0])
                    new_env = deepcopy(self.venv.envs[0])
                    envs.append(new_env)

                self.venv.envs = envs
                ID = [[0] for i in range(size_search)]

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