import gym
from gym import spaces
import numpy as np
from numba import jit

#@jit(nopython=True)
def chooseAction(jobToChoose, job_times, jobsState, \
                 maxJobLength, smaxSpan, \
                 job_early_start_time,precedence,\
                 machine_utilization, jobDone, jobsToProcess, placement, order):

    # operation index
    jobOrder = jobsState[jobToChoose] # note: maximum = num_ops
    # print (jobOrder)
    maxSpan = smaxSpan # current truncated makespan
    # if index is out of bound, job was already finished.
    if jobOrder ==  maxJobLength:  
        reward = -1  # masking invalid action would not trigger this if statement
        pass
    else:        
        job_early_start =  job_early_start_time[jobToChoose] # how long does it take to start the operations for the chosen job
        
        job_time =  job_times[jobToChoose][jobOrder] # operation time for the chosen job
        job_machine_placement =  precedence[jobToChoose][jobOrder] # which machine to perform the operation for the chosen job
        
        
        placement[job_machine_placement,order[job_machine_placement],:] =([jobToChoose, jobOrder])
        order[job_machine_placement] += 1

        machine_current_utilization = machine_utilization[job_machine_placement] # how long this machine has been running

        if job_early_start > machine_current_utilization:
            machine_current_utilization = job_early_start + job_time
        else:
            machine_current_utilization +=  job_time

        job_early_start_time[jobToChoose] = machine_current_utilization   

        machine_utilization[job_machine_placement] = machine_current_utilization
        
        if machine_utilization[job_machine_placement] > maxSpan:
            maxSpan = machine_utilization[job_machine_placement]

        jobsState[jobToChoose]+=1
        if  jobOrder+1 == maxJobLength:
            jobsToProcess -= 1
            jobDone[jobToChoose] = 1 # doesn't really use this?
            
        reward = float(smaxSpan - maxSpan)

    done = 1
    if  jobsToProcess > 0:
        done = 0
    return jobsToProcess, maxSpan, reward, done, job_early_start_time, machine_utilization, jobsState, jobDone, placement

class JobShopGymEnv(gym.Env):
    
    metadata = {'render.modes':['console']}
    
    def __init__(self, maxJobs = 10, maxJobLength=10, n_machines = 10):
        super(JobShopGymEnv, self).__init__()
        self.maxJobs=maxJobs
        self.maxJobLength=maxJobLength
        self.n_machines = n_machines
        self.n_actions = self.maxJobs
        self.action_space = spaces.Discrete(self.n_actions)
        self.NONE_MACHINE = n_machines
        precedenceStates=[[self.n_machines+1]*self.maxJobLength]*self.maxJobs
        
        self.observation_space=spaces.Dict({
                # this will keep track of when a new job can start on this machine
                'machine_utilization': spaces.Box(low=0,high=(self.maxJobs*self.maxJobLength),shape=(self.n_machines,))   ,
                # for each job this tell you want it can start running
                'job_early_start_time': spaces.Box(low=0,high=(self.maxJobs*self.maxJobLength),shape=(self.maxJobs,))   ,
                # what are the running times for given tasks for given jobs
                'job_times': spaces.Box(low=0,high=1,shape=(self.maxJobs,self.maxJobLength))   ,
                # precedence matrix
                'precedence': spaces.MultiDiscrete(precedenceStates) })
        
        self.precedenceInp=[]
        self.time_preInp=[]
        self.jobsState = np.zeros(self.maxJobs,dtype=np.int64)
        self.maxSpan = 0.0
        self.jobsToProcess = self.maxJobs
        self.jobDone=np.zeros(self.maxJobs)
        self.info = {}

        self.placement = np.zeros((self.n_machines,self.maxJobs,2), dtype=int)
        self.order = np.zeros((self.n_machines), dtype=int)
        
    def setGame(self,precedence,time_pre):
        self.precedenceInp=precedence
        self.time_preInp=np.array(time_pre,dtype=np.float32)
        self.maxSpan = 0.0
        self.jobsToProcess = self.maxJobs    
        self.jobDone=np.zeros(self.maxJobs)
        
    def reset(self):
        state = self.observation_space.sample()

        state['job_times']=np.array(self.time_preInp)+0.0
        state['machine_utilization']=state['machine_utilization']*0
        state['job_early_start_time']=state['job_early_start_time']*0
        state['precedence']=np.array(self.precedenceInp)+0
        
        self.jobsState = np.zeros(self.maxJobs,dtype=np.int64) # keep track of how many tasks for given job has been scheduled
        self.state = state
        self.maxSpan = 0.0
        self.jobsToProcess = self.maxJobs
        self.jobDone=np.zeros(self.maxJobs)

        self.placement = np.zeros((self.n_machines,self.maxJobs,2), dtype=int)
        self.order = np.zeros((self.n_machines), dtype=int)
        
        return state
    
    def faststep(self, jobToChoose):
        jobsToProcess, maxSpan, reward, done, \
            job_early_start_time, machine_utilization, \
                jobsState, jobDone, placement = chooseAction(jobToChoose, self.state['job_times'], 
                                                             self.jobsState,self.maxJobLength, self.maxSpan,
                                                             self.state['job_early_start_time'], 
                                                             self.state['precedence'],
                                                             self.state['machine_utilization'], 
                                                             self.jobDone, self.jobsToProcess, self.placement, 
                                                             self.order)
        self.maxSpan = maxSpan
        self.jobsToProcess = jobsToProcess
        self.state['job_early_start_time'] = job_early_start_time
        self.state['machine_utilization'] = machine_utilization
        self.jobsState = jobsState
        self.jobDone = jobDone

        self.placement = placement 
       
        
        return self.state, reward, done==1, self.info
    
    def render(self,mode='console'):
        if mode != 'console':
            raise NotImplementedError()
    
    def close(self):
        pass
        
        
        
        
        
        
        