#------------------------------------------------------------------------
# > Intros:
# This is the implementation of LR policies, including 'SGDR' and 'step 
# with decay'. 
#
# For more details about 'SGDR', here is a reference:
# SGDR: Stochastic Gradient Descent with Warm Restarts
# [https://arxiv.org/abs/1608.03983]
#
# > Author: Wenhao Niu
# > Date: Aug 26, 2018
#------------------------------------------------------------------------


import yaml
import numpy as np


class LrPolicy(object):
    '''
    > Intros:
    ---------------------------------------------------------------------------------
    This is a class of LR policies, you can choose either 'SGDR' or 'step with decay'
    by setting proper params.

    > Args:
    ---------------------------------------------------------------------------------
    policy:       Poloicy name. You can choose either 'sgdr' or 'step'. Inputs beyond 
                  these two words would raise error.
    max_iter:     Maximum iterations of your training. You can either preset in the 
                  config file or set when initializing `LrPolicy` objects.

    For more info of parameters, you can read reference paper or check config file.

    > Example:
    ---------------------------------------------------------------------------------
    >>> LR = LrPolicy('sgdr',max_iter)
    >>> [training Loop]:
    >>>     cur_lr = LR.get_lr_at_iter(cur_iter)
    '''
    def __init__(self, policy, max_iter=None):

        with open('config.yaml','r') as f:
            cfg = yaml.load(f)
            self.base_lr = cfg['BASE_LR']
            self.max_iter = cfg['MAX_ITER']
            self.policy = policy
            
            # sgdr paras
            self.tmul = cfg['SGDR']['TMUL']
            self.restart_times = cfg['SGDR']['RESTART_TIMES']
            self.lr_max = cfg['BASE_LR']
            self.lr_min = self.lr_max * cfg['SGDR']['LR_MIN_RATIO']
            
            # step paras
            self.decay_ratio = cfg['STEP']['DECAY_RATIO']
            self.step = cfg['STEP']['STEP_SIZE']

        if max_iter:
            self.max_iter = max_iter

    

    #------------------------------------------------------------------
    # Lr policy functions
    #------------------------------------------------------------------

    def sgdr_func(self,cur_iter):
        iter_from_restart, T = self.get_T(cur_iter)
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1. + np.cos(float(iter_from_restart) / float(T) * np.pi))
        return lr

    def step(self,cur_iter):
        lr = self.base_lr * self.decay_ratio ** (cur_iter // self.step)
        return lr

    #------------------------------------------------------------------
    # Helper
    #------------------------------------------------------------------
    
    def get_T(self,cur_iter):
        T = int(np.ceil(self.max_iter * (self.tmul - 1.) / (self.tmul ** self.restart_times - 1.) ))

        while cur_iter >= T:
            cur_iter -= T
            T *= self.tmul

        return cur_iter, T

    #------------------------------------------------------------------
    # API
    #------------------------------------------------------------------

    def get_lr_at_iter(self,it):
        
        if self.policy == 'sgdr':
            return self.sgdr_func(it)

        elif self.policy == 'step':
            return self.step_func(it)

        else:
            raise Exception('Wrong key words for lr policy')

if __name__=="__main__":
    LR = LrPolicy('sgdr')
    for i in range(LR.max_iter):
        print(LR.get_lr_at_iter(i))
