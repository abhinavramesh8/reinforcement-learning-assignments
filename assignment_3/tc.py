import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        n_tiles = np.ceil((state_high - state_low) / tile_width).astype(int) + 1
        self.tile_weight = np.zeros((num_tilings, *n_tiles))
        
        offset = np.outer(np.arange(num_tilings), tile_width) / num_tilings
        self.start_index = state_low - offset
        
        self.n_tilings = num_tilings
        self.tile_width = tile_width
    
    def __get_active_indices(self, s):
        idx = ((s - self.start_index) / self.tile_width).astype(int)
        return [(i,*idx[i]) for i in range(self.n_tilings)]

    def __call__(self,s, active_indices=None):
        if active_indices is None:
            active_indices = self.__get_active_indices(s)
        return np.sum([self.tile_weight[i] for i in active_indices])
        
    def update(self,alpha,G,s_tau):
        active_indices = self.__get_active_indices(s_tau)
        v = self(s_tau, active_indices)
        weight_update = alpha * (G - v)
        for i in active_indices:
            self.tile_weight[i] += weight_update
        return None
