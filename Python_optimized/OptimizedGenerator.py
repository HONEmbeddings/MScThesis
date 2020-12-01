import numba
from numba import int32, float64
from numba.experimental import jitclass

import numpy as np
import pandas as pd

# do not import ABCHigherOrderPathGenerator

# The simulation of random walks and the calculation of the visitation probabilities originally describe
# the transition probabilities via tuple of nodes. This requires dictionary look-ups.

# To improve perforance, we first enumerate the possible states of the Markov chain and use these indices
# to describe the transition probabilities. Hence, for each and transition from (s1,...,sk) to t, we need:
# - the probability P( t | (s1,...,sk))
# - the index ot the next node t
# - the index of the next Markov state (s2,...,sk, t)
# This renders any dictionary lookups unnecessary (besides identifying the start-index of a random walk).

# Numba is a JIT compiler for Python and Numpy, which greatly improves the speed (especially of for loops)
# but supports only a subset of the Python code. Luckily, using indices instead of tuples of anything 
# simplifies the data types used - only int32 and float 64.
# The initialization of OptimizedGenerator from a HigherOrderPathGenerator is outside the class in 
# create_OptimizedGenerator, due to restrictions by Numba.
# Moreover, Numba does not support individual numpy random generators; see
# https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#random .
# Therefore, generate the random values outside the OptimizedGenerator; see HigherOrderPathGenerator.

spec = [
    ('_num_target_nodes', int32),
    ('_num_source_paths', int32),
    ('_transition_offsets', int32[:]),
    ('_transition_next_node', int32[:]),
    ('_transition_next_path', int32[:]),
    ('_transition_probs', float64[:]),
    ('_transition_probs_cumulated', float64[:]),
]
@jitclass(spec)
class OptimizedGenerator(object):
    def __init__(self, num_target_nodes, transition_offsets, transition_next_node, transition_next_path, transition_probs, transition_probs_cumulated):
        self._num_target_nodes = num_target_nodes
        self._num_source_paths = len(transition_offsets) - 2
        self._transition_offsets = transition_offsets
        self._transition_next_node = transition_next_node
        self._transition_next_path = transition_next_path
        self._transition_probs = transition_probs
        self._transition_probs_cumulated = transition_probs_cumulated

    def _random_step_index(self, start_idx: int, random_value: float) -> int:
        """Simulates a random step.
        
        Parameters
        ----------
        start_idx : int
            Index of the start of the random step, see generator._source_paths_dict
        
        random_value : float, >=0, <=1
            Random value
        
        Returns
        -------
        int
            Index of the next node (in self._transition_next_node) or next path (in self._transition_next_path)
        
        Example
        -------
        gen = create_lattice_2nd_order_dynamic(4, 0.5)
        source_paths = gen.source_paths
        ogen = create_OptimizedGenerator(gen)
        start_idx = np.random.randint(len(source_paths))
        start_path = source_paths[start_idx]
        j = ogen._random_step_index(start_idx, np.random.random())
        next_node_idx = ogen._transition_next_node[j]
        next_node = gen.target_nodes[next_node_idx]
        next_path_idx = ogen._transition_next_path[j]
        next_path = gen.source_paths[next_path_idx]
        print(start_path, '->', next_node, '=>', next_path)
        """
        offset = self._transition_offsets[start_idx]
        offset_next = self._transition_offsets[start_idx + 1]
        return offset + np.searchsorted(self._transition_probs_cumulated[offset:offset_next], random_value, 'left')
        #return bisect.bisect_left(self._transition_probs_cumulated, random_value, lo=offset, hi=offset_next-1)
    
    def random_walks(self, start_idxs: np.ndarray, random_values: np.ndarray):
        """Simulates random walks with given start indices.
        
        Parameters
        ----------
        start_idxs : np.ndarray
            Indices of the start of the random step, see generator._source_paths_dict
        
        random_values : 
            Random values in [0,1] for each step to simulate
        
        Returns
        -------
        np.ndarray
            Indices (referring to generator.target_nodes) of the nodes visited. 
        """
        res = np.zeros(random_values.shape, int32)
        for i,start_idx in enumerate(start_idxs):
            for j in range(random_values.shape[1]):
                k = self._random_step_index(start_idx, random_values[i,j])
                res[i,j] = self._transition_next_node[k]
                start_idx = self._transition_next_path[k]
        return res
    
    def walk_probs(self, num_steps: int = 1, pairwise: bool = True, aggregate_steps: bool=True, step_factor: float = 1.0) -> np.ndarray:
        """Calculate the visitation probabilities of random walks
        
        Parameters
        ----------
        num_steps : int, >0
            Number of steps
        
        pairwise : bool, optional (default = True)
            Calculating random walk visitation probabilities starting with any source path in self.source_paths (False) 
            or only with source paths of length one (True).
        
        aggregate_steps : bool, optional (default = True)
            Calculate the random walk visitation probabilities (True) or the multi-step visitation probabilities (False).
            
        step_factor : float, >0, <=1, optional (default = 1.0)
            When aggregating multi-step visitation probabilities, each step is weigthed with (1/num_steps) if step_factor=1.0,
            or proportional to step_factor**step, respectively.
            The latter is related to personalized page rank (PPR) for large num_steps.
                    
        Returns 
        -------
        np.ndarray
            If aggregate_steps = False, the result is a matrix res[start_idx, next_node_idx, step-1] containing the multi-step 
            transition probabilities. Otherwise, the third dimension is ignored and res[start_idx, next_node_idx, 0] contain the
            visitation frequencies (or PPR, if step_factor < 1).
        """
        # this assumes that start_nodes is sorted by len
        num_start = self._num_target_nodes if pairwise else self._num_source_paths
        res = np.zeros(shape = (num_start, self._num_target_nodes, 1 if aggregate_steps else num_steps))

        if aggregate_steps:
            if step_factor < 1 and num_steps > 1:
                init_prob = (1.0 - step_factor**(num_steps+1)) / (1.0 - step_factor)
            else:
                init_prob = 1.0 / num_steps
        else:
            init_prob = 1.0
            step_factor = 1.0 # ignore
            
        for start_idx in range(num_start):
            #paths = { start_idx: init_prob}
            paths = numba.typed.Dict.empty(key_type=int32, value_type=float64)
            paths[start_idx] = init_prob
            for step0 in range(num_steps):
                if aggregate_steps: step0 = 0
                #paths_new = defaultdict(float)
                paths_new = numba.typed.Dict.empty(key_type=int32, value_type=float64)
                for source_idx, source_prob in paths.items():
                    source_prob *= step_factor
                    offset =  self._transition_offsets[source_idx]
                    offset_next =  self._transition_offsets[source_idx + 1]
                    for i in range(offset,offset_next):
                        prob = self._transition_probs[i]
                        prob_new = source_prob * prob
                        next_node = self._transition_next_node[i]
                        res[start_idx, next_node, step0] += prob_new
                        next_path = self._transition_next_path[i]
                        paths_new[next_path] = paths_new.get(next_path, 0.0) + prob_new
                paths = paths_new
        return res
 
def create_OptimizedGenerator(gen : "ABCHigherOrderPathGenerator"):
    # the correctness of walk_probs relies on gen.source_paths is sorted by len and
    # assert () not in gen.source_paths, 'empty path must not be in source_paths'
    # assert () in gen._source_paths_dict, 'empty path must be in source_paths'
    # for i,source_path in gen.source_paths:
    #     assert gen._source_paths_dict[source_path] == i,'sort order'
    # assert gen._source_paths_dict[()] == len(gen.source_paths), 'empty path must have the last index'

    #source_paths = gen.source_paths # does not contain empty path
    source_paths_dict = gen._source_paths_dict
    target_nodes = gen.target_nodes
    target_nodes_dict = gen._target_nodes_dict

    transition_offsets = np.zeros(len(source_paths_dict) + 1, dtype=np.int32)
    # calculate offsets
    offset = 0
    for source_path, source_idx in source_paths_dict.items():
        transition_offsets[source_idx] = offset
        offset += len(list(gen.transition_probs(start=source_path)))
    transition_offsets[-1]=offset
    # calculate probs
    transition_next_node = np.zeros(offset, dtype=np.int32)
    transition_next_path = np.zeros(offset, dtype=np.int32)
    transition_probs = np.zeros(offset, dtype=np.float64)
    transition_probs_cumulated = np.zeros(offset, dtype=np.float64)
    for source_path,source_idx in source_paths_dict.items():
        offset = transition_offsets[source_idx]
        offset_next = transition_offsets[source_idx + 1]
        num_transitions = offset_next - offset
        next_nodes = np.zeros(num_transitions, np.int32)
        next_paths = np.zeros(num_transitions, np.int32)
        probs = np.zeros(num_transitions, np.float64)
        for i,(_,next_node,prob) in enumerate(gen.transition_probs(start=source_path)):
            probs[i] = prob
            next_nodes[i] = target_nodes_dict[next_node]
            next_path_name = gen.find_rule_key((*source_path, next_node))
            next_paths[i] = source_paths_dict[next_path_name]
        transition_next_node[offset:offset_next] = next_nodes
        transition_next_path[offset:offset_next] = next_paths
        transition_probs[offset:offset_next] = probs
        probs_cumulated = np.cumsum(probs)
        transition_probs_cumulated[offset:offset_next] = probs_cumulated / max(probs_cumulated[-1], 1e-20)
    num_target_nodes = len(target_nodes)
    return OptimizedGenerator(num_target_nodes, transition_offsets, transition_next_node, transition_next_path, transition_probs, transition_probs_cumulated)

def debug_OptimizedGenerator(gen : "ABCHigherOrderPathGenerator", ogen : OptimizedGenerator) -> pd.DataFrame:
        "Debug the probabilities"
        source_paths = gen.source_paths
        target_nodes = gen.target_nodes
        
        source_paths_idx = list()
        for i in range(len(ogen._transition_offsets) - 1):
            n = ogen._transition_offsets[i+1] - ogen._transition_offsets[i]
            source_paths_idx.extend(np.full(n, i))
        data = dict(
            source_path = list(map(lambda i:source_paths[i] if i<len(source_paths) else None, source_paths_idx)),
            next_node = list(map(lambda i:target_nodes[i], ogen._transition_next_node)),
            next_path = list(map(lambda i:source_paths[i], ogen._transition_next_path)),
            prob = ogen._transition_probs,
            cumulated_prob = ogen._transition_probs_cumulated,
            source_paths_idx = source_paths_idx,
            next_node_idx = ogen._transition_next_node,
            next_path_idx = ogen._transition_next_path,
        )
        return pd.DataFrame(data)