# parent class for state representation

import torch
torch.set_default_dtype(torch.float64)

import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .Set2Set_test import Set2Set

class StateLSTM(nn.Module):
    def __init__(self, _embeddedDim, _jobs, _ops, _macs, device='cuda:0'):
        super(StateLSTM, self).__init__()
        self.embeddedDim = _embeddedDim
        self.jobs = _jobs
        self.ops = _ops
        self.macs = _macs
        self.device = device

        # instance representation layers
        # self.machinesEmbedding = nn.Linear(self.embeddedDim,self.macs+1,bias = False)
        self.machinesEmbedding = nn.Linear(1, self.embeddedDim) # size agnostic
        self.jobTimeEmbedding = nn.Linear(1, self.embeddedDim)
#         self.jobTimeEmbedding2 = nn.Linear(self.embeddedDim, self.embeddedDim) # redundant?
        self.sequenceLSTM = nn.LSTM(2*self.embeddedDim, self.embeddedDim, batch_first=True) # set batch size as 1st Dim
    
        # dynamic representation layers
        self.jobStartTimeEmbed = nn.Linear(1, self.embeddedDim)
        self.machineTimeEmbed = nn.Linear(1, self.embeddedDim)
        
        # activation function
        self.activation = F.leaky_relu

        # Set2Set function to get inter Job Embeddings
        self.interJobEmbedding = Set2Set(3*self.embeddedDim, 1, 1)
        
    def numel(self):
        
        return np.sum([torch.numel(w) for w in self.parameters()])
       
    def instanceEmbedding(self,State):
        # input: State - OrderedDict
        
        
        #
        # instance representation
        #
        
        Job_times = torch.tensor(State['job_times'],device=self.device,dtype=torch.float64)
        Precedences = torch.tensor(State['precedence'],device=self.device,dtype=torch.int64)
        BS = Precedences.shape[0]
            
        # augment input to add individual job terminal state
        Precedences_extra = self.macs*torch.ones(BS,self.jobs,1,device=self.device,dtype=torch.int64)
        Job_times_extra = torch.zeros(BS,self.jobs,1,device=self.device,dtype=torch.float64)
               
        # augmented input
        Precedences = torch.cat((Precedences,Precedences_extra),dim=2) # BS, num_jobs, num_ops+1
        Job_times = torch.cat((Job_times,Job_times_extra),dim=2) # BS, num_jobs, num_ops+1
        
        # embedding job times                        #? are activation and 2nd layer necessary here?
#         Job_times = self.jobTimeEmbedding2(self.activation(self.jobTimeEmbedding(Job_times.unsqueeze(3))))
        Job_times = self.jobTimeEmbedding(Job_times.unsqueeze(3))
        # embedding precedence 
        # Precedences = self.machinesEmbedding.weight[[Precedences]] 
        Precedences_float = Precedences.unsqueeze(3).to(dtype=torch.float64)
        Precedences = self.machinesEmbedding(Precedences_float) # size agnostic
        # concat embeded precedence and job time - input has to be a 3D tensor: batch, seq, feature
        PrecedenceTime =torch.cat((Precedences,Job_times),dim=3)
        PrecedenceTime = PrecedenceTime.reshape(BS*self.jobs,self.ops+1,-1) 
        
        # LSTM embedding 
        JobEmbeddings,_ = self.sequenceLSTM(torch.flip(PrecedenceTime,[1])) # did a flip for reverse sequence
        JobEmbeddings = JobEmbeddings.reshape(BS,self.jobs,self.ops+1,-1) # shape - BS, num_jobs, num_ops+1, embed_dim

        return JobEmbeddings
    
    def dynamicEmbedding(self,State,JobEmbeddings):
        
        # option 1:
        # match up machine utilization with job early start time by assignment 1 on 1
        
        # 
        # dynamic representation
        #
        
        Job_early_start_time = torch.tensor(State['job_early_start_time'],dtype=torch.float64,device=self.device).unsqueeze(2)
        BS = Job_early_start_time.shape[0]
        
        Machine_utilization = torch.tensor(State['machine_utilization'],dtype=torch.float64,device=self.device).unsqueeze(2)
        # add extra machine for when job is finished
        Machine_utilization_extra = torch.zeros(BS,1,1,dtype=torch.float64,device=self.device)
        Machine_utilization = torch.cat((Machine_utilization,Machine_utilization_extra),dim=1) #BS, mac+1, 1 
        
        # embedding 
        Job_early_start_time = self.jobStartTimeEmbed(Job_early_start_time) # BS, num_jobs, emded_dim
        Machine_utilization = self.machineTimeEmbed(Machine_utilization) # BS, num_mac+1, emded_dim
          
        BSID = [[i] for i in range(BS)]
        JobID = [[i for i in range(self.jobs)] for j in range(BS)]
             
        JobEmbeddings = JobEmbeddings[BSID,JobID,self.ops-State['job_state'],:]
        
        #
        # select machine for ops by job
        #
        
        # editing precedence
        MacID = np.concatenate((State['precedence'], self.macs*np.ones([BS,self.jobs,1])), axis=2)
        MacID = MacID[BSID,JobID,State['job_state']]
        
        Machine_utilization = Machine_utilization[BSID,MacID,:]
        
        stateEmbeded = torch.cat((JobEmbeddings,Machine_utilization,Job_early_start_time),dim=2)

        stateEmbeded = self.interJobEmbedding(stateEmbeded)
                                   
        return stateEmbeded
