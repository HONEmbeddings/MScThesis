from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Iterator, Tuple, Optional # https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
from collections import defaultdict, Counter
import math
import numpy as np
import pandas as pd
#import random
import bisect
from EmbeddingData import EmbeddingData
from OptimizedGenerator import OptimizedGenerator, create_OptimizedGenerator
import warnings

class ABCHigherOrderPathGenerator(ABC):
    "Generator for higher-order paths (abstract) - storage of rules is not implemented here"
    def __init__(self, node_sort_key=lambda node: node, id: Optional[str] = None, config: Dict[str,Any] = dict(),
            create_EmbeddingData = None):
        self._rules_cumulated = None
        self._node_sort_key = node_sort_key
        self._id = type(self).__name__ if id is None else id
        self._source_path_metadata = dict() # Dict[str, Dict[Tuple[Any,...],Any]]
        self._target_node_metadata = dict() # Dict[str, Dict[Any,Any]]
        self._config = config
        self._frozen = False
        if create_EmbeddingData is None:
            # Class EmbeddingData provides minimal functionality, as it avoids dependency on visualization.
            # The create_EmbeddingData argument allows for customizing the embedding data;
            # see also create_EmbeddingData_Vis from Visualizations.py
            create_EmbeddingData = lambda *args, **kwargs: EmbeddingData(*args, **kwargs)
        self._create_EmbeddingData = create_EmbeddingData 

    # metadata will be used in the visualizations
    def add_metadata(self, name: str, data: Dict[Any,Any], use_last: bool = True, check: bool = True):
        """
        Add metadata describing nodes for both target_nodes and source_paths.
        For the latter use either the first or last node of the path.
        """
        self.add_target_node_metadata(name, data, check)
        index = -1 if use_last else 0
        paths_data = { key: data.get(key[index], None) for key in self.source_paths }
        self.add_source_path_metadata(name, paths_data, check=False)

    def add_source_path_metadata(self, name: str, data: Dict[Tuple[Any,...],Any], check: bool = True):
        "Add metadata describing the source_paths"
        if check:
            assert len(set(data.keys()).difference(self.source_paths))==0, 'metadata "%s" contains invalid keys' % name
        self._source_path_metadata[name] = data

    def add_target_node_metadata(self, name: str, data: Dict[Any,Any], check: bool = True):
        "Add metadata describing the target_nodes"
        if check:
            assert len(set(data.keys()).difference(self.target_nodes))==0, 'metadata "%s" contains invalid keys' % name
        self._target_node_metadata[name] = data

    def create_source_embedding_data(self, emb, dimension: int, pairwise: bool = False) -> EmbeddingData:
        # factory method used in Embedding.py
        keys = self.source_paths_len1 if pairwise else self.source_paths
        keys_dict = self._source_paths_dict
        return self._create_EmbeddingData(emb, dimension, keys=keys, keys_dict=keys_dict,  
            use_source=True, metadata = self._source_path_metadata)

    def create_target_embedding_data(self, emb, dimension: int) -> EmbeddingData:
        # factory method used in Embedding.py
        keys = self.target_nodes
        keys_dict = self._target_nodes_dict
        return self._create_EmbeddingData(emb, dimension, keys=keys, keys_dict=keys_dict,  
            use_source=False, metadata = self._target_node_metadata)

    @property
    @abstractmethod
    def rule_keys(self) -> Iterator[Tuple[Any,...]]:
        pass

    @abstractmethod
    def find_rule_key(self, start: Tuple[Any,...]) -> Tuple[Any,...]:
        pass

    @property
    @abstractmethod
    def max_rule_key_length(self) -> int:
        pass

    @property
    def source_paths(self) -> List[Tuple[Any,...]]:
        "returns a sorted list of source paths of the transition robabilities (empty key is removed)"
        assert self._frozen, 'please run freeze_rules() before accessing source_paths'
        return self._source_paths
        # sort_key = lambda path:(len(path), *map(self._node_sort_key, path))
        # return sorted({ key for key in self.rule_keys if len(key) >= 1 }, key=sort_key)
    
    @property
    def source_paths_len1(self) -> List[Tuple[Any,...]]:
        "returns a sorted list of source paths of length 1 of the transition robabilities"
        assert self._frozen, 'please run freeze_rules() before accessing source_paths_len1'
        return self._source_paths_len1
        # sort_key = lambda path:self._node_sort_key(path[0])
        # #return sorted({ (v,) for key in self.rule_keys for v in key }, key=sort_key) # safe & slower
        # return sorted({ key for key in self.rule_keys if len(key) == 1 }, key=sort_key)

    @property
    def target_nodes(self) -> List[Tuple[Any,...]]:
        "returns a sorted list target nodes of the transition robabilities"
        assert self._frozen, 'please run freeze_rules() before accessing target_nodes'
        return self._target_nodes
        # #return sorted({ next_node for key in self.rule_keys for _, next_node, _ in self.transition_probs(key) }) # safe & slower
        # return sorted({ key[0] for key in self.rule_keys if len(key) == 1 }, key=self._node_sort_key) # this should also work, as otherwise random walks are broken.

    def freeze_rules(self, check=False):
        sort_key = lambda path:(len(path), *map(self._node_sort_key, path))
        self._source_paths = sorted({ key for key in self.rule_keys if len(key) >= 1 }, key=sort_key)
        self._source_paths_dict = { p:i for i,p in enumerate(self._source_paths) } # does not contain empty path
        self._source_paths_dict[()] = len(self._source_paths) # add empty path to dictionary
        self._source_paths_len1 = list(key for key in self._source_paths if len(key)==1)
        self._target_nodes = sorted({ key[0] for key in self.rule_keys if len(key) == 1 }, key=self._node_sort_key)
        self._target_nodes_dict = { p:i for i,p in enumerate(self._target_nodes) }
        if check:
            # target_nodes was constructed from source_paths to include nodes with in-degree zero.
            # This check detects if any target node of a transition probability is missing in target_nodes,
            # which will cause a runtim error in create_OptimizedGenerator, below.
            for key in self.rule_keys:
                for _,node,_ in self.transition_probs(start=key):
                    assert node in self._target_nodes_dict, f"Inconsistent rules: target-node '{node}' lacks a FON rule"
        self._rules_cumulated = None  # todo
        self._frozen = True
        self._optimized_gen = create_OptimizedGenerator(self) # If Error set check=True

    @abstractmethod
    def transition_probs(self, start: Tuple[Any,...]) -> Iterator[Tuple[Tuple[Any,...],Any,float]]:
        pass

    def check_transition_probs(self, tol=1e-14) -> Iterator[Tuple[Tuple[Any,...],float]]:
        "Verify that the transition probabilities sum up to one (with some numerical tolerance)."
        for key in self.rule_keys:
            total_prob = sum(prob for _,_,prob in self.transition_probs(key))
            if abs(total_prob-1)>tol:
                yield (key, total_prob)

    def path_probs(self, start: Tuple[Any,...], num_steps: int = 1) -> Iterator[Tuple[Tuple[Any,...],float]]:
        "Conditional probabilities of all paths starting with some given sequence of nodes."
        if num_steps <= 0:
            yield (start, 1.0)
        else: #elif num_steps >= 1:
            for outer_path, outer_prob in self.path_probs(start, num_steps-1):
                for _, next_node, prob in self.transition_probs(outer_path):
                    yield ((*outer_path, next_node), outer_prob*prob)
    
    def marginal_path_probs(self, start: Tuple[Any,...], num_steps: int = 1, projection: Callable[[Tuple[Any,...]] ,Any] = lambda x:x[-1]) -> Dict[Any,float]:
        "Calculates the marginal probabilities of path_probs. May be used to calculate k-step transition probabilities."
        probs = defaultdict(float)
        for path,prob in self.path_probs(start,num_steps):
            key = projection(path)
            probs[key]+=prob
        return probs

    def verify_stationarity(self, order=2, tol=1e-14) -> Iterator[Tuple[Any,float,float,float]]:
        """
        Verify that the stationary distribution is indeed stationary with respect to the first order transition probabilities.
        
        The (1st order) stationary distribution is stored as the transition probabilities of an empty start path.
        """
        mprobs = self.marginal_path_probs(start=(), num_steps=order+1, projection=lambda x:x[-1]) # node-distribution after a few steps
        for _,node,prob in self.transition_probs(start=()): # stationary distribution
            mprob = mprobs[node]
            if abs(prob - mprob) > tol:
                yield (node, prob, mprob, prob-mprob)
    
    @property
    def rules_cumulated(self):
        "Cumulates the transition probabilities for random.choices"
        if self._rules_cumulated is None:
            rules2 = dict()
            for start in self.rule_keys:
                next_nodes = list()
                cum_weights = list()
                cum_prob = 0.0
                for _, next_node, prob in self.transition_probs(start):
                    cum_prob += prob
                    next_nodes.append(next_node)
                    cum_weights.append(cum_prob)
                rules2[start] = (next_nodes, cum_weights) 
            self._rules_cumulated = rules2
        return self._rules_cumulated

    def random_step(self, start: Tuple[Any,...], size=None, rng: Optional[np.random.Generator] = None) -> Any:
        if rng is None:
            rng = np.random.default_rng()
        rule_key = self.find_rule_key(start)
        if rule_key == () and start != (): # or assert start is a tuple
            print('could not find a key matching %s' % start)
        next_nodes,cum_weights = self.rules_cumulated[rule_key]
        if size is None: # return scalar
            x = rng.random(1)
            # return random.choices(next_nodes, cum_weights=cum_weights, k=1)[0] # fine, but wrong RNG
            return next_nodes[bisect.bisect_left(cum_weights, x[0], hi=len(cum_weights)-1)]
        else: # return list
            x = rng.random(size)
            f = lambda y:next_nodes[bisect.bisect_left(cum_weights, y, hi=len(cum_weights)-1)]
            return list(map(f, x))
    
    def random_walk(self, start: Tuple[Any,...], num_steps:int = 1, rng: Optional[np.random.Generator] = None) -> Tuple[Any,...]:
        if rng is None:
            rng = np.random.default_rng()
        walk = start
        for _ in range(num_steps):
            next_node = self.random_step(walk, size=None, rng=rng)
            walk = (*walk,next_node)
        return walk

    def to_FON(self, id: Optional[str]  =None) -> "HigherOrderPathGenerator":
        "Return a HigherOrderPathGenerator containing only first-order rules (and the stationary distribution)"
        if id is None:
            id = self._id + ' (FON)'
        fon = HigherOrderPathGenerator(node_sort_key=self._node_sort_key, id=id, config=self._config, 
            create_EmbeddingData=self._create_EmbeddingData)
        if hasattr(self, 'creator'): # see Lattice2D_2nd_order_dynamic & required for Lattice2D_EmbeddingView
            fon.creator = self.creator
        for key in self.rule_keys:
            if len(key)>1:
                continue
            for start, next_node, prob in self.transition_probs(key):
                fon.add_rule(start, next_node, prob)
        fon.freeze_rules()
        for name, data in self._source_path_metadata.items():
            fon.add_source_path_metadata(name, { key: value for key,value in data.items() if len(key)==1 }, check=True)
        fon._target_node_metadata = dict(self._target_node_metadata)
        return fon

    # optimized calculations
    def random_walks(self, start_list, num_steps: int=1, rng: Optional[np.random.Generator] = None, include_start: bool=True):
        if rng is None:
            rng = np.random.default_rng()
        target_nodes = self.target_nodes
        #start_idxs = np.array(list(map(lambda start: self._source_paths_dict[self.find_rule_key(start)], start_list)), dtype=np.int32)
        start_idxs = np.array([self._source_paths_dict[self.find_rule_key(start)] for start in start_list], dtype=np.int32)
        random_values = rng.random(size=(len(start_idxs), num_steps), dtype=np.float64)
        res_idx = self._optimized_gen.random_walks(start_idxs, random_values)
        res = list()
        for i,start in enumerate(start_list):
            nodes = tuple(map(target_nodes.__getitem__, res_idx[i]))
            if include_start:
                res.append((*start, *nodes))
            else:
                res.append(nodes)
        return res

    def walk_probs(self, num_steps: int = 1, pairwise: bool = True, aggregate_steps: bool=True, step_factor: float = 1.0) -> np.ndarray:
        assert self._frozen, 'please run freeze_rules() before accessing walk_probs'
        res = self._optimized_gen.walk_probs(num_steps, pairwise, aggregate_steps, step_factor)
        return res.reshape(res.shape[:-1]) if aggregate_steps else res
    
    def walk_probs_df(self, num_steps: int = 1, pairwise: bool = True, step_factor: float = 1.0) -> pd.DataFrame:
        "Convert the output of walk_probs() to a pandas DataFrame for debugging."
        res = self.walk_probs(num_steps=num_steps, pairwise=pairwise, aggregate_steps=True, step_factor=step_factor)
        indices = list(map(str, self.source_paths[:res.shape[0]]))
        columns = list(map(str, self.target_nodes))
        return pd.DataFrame(res, index=indices, columns=columns)

    @property
    def config(self):
        "get configuration"
        cfg=dict(init_class=self.__class__.__name__, init_id=self._id)
        cfg.update(self._config)
        return cfg

    def write_config(self, file_object, comment: str = '', prefix: str = '', sep: str ='\t'):
        if comment != '':
            file_object.write(prefix + comment + '\n' + prefix + '\n')
        file_object.write(prefix + 'HigherOrderPathGenerator:\n')
        for k,v in self.config.items():
            file_object.write(prefix + '%s%s%s\n' % (k,sep,v))

class HigherOrderPathGenerator(ABCHigherOrderPathGenerator):
    "Generator for higher-order paths"
    def __init__(self, node_sort_key=lambda node: node, id: Optional[str] = None, config: Dict[str,Any] = dict(),
            create_EmbeddingData = None):
        super().__init__(node_sort_key, id, config, create_EmbeddingData)
        self.rules = defaultdict(dict)
        self._max_rule_key_length = 1 # find_rule_key relies on _max_rule_key_length > 0

    @property
    def max_rule_key_length(self) -> int:
        return self._max_rule_key_length

    def clear_rules(self):
        "Clear the rules dictionary (incl. clean up)"
        self._max_rule_key_length = 1
        self.rules.clear()
        self._frozen = False
        self._rules_cumulated = None

    def add_rule(self, start: Tuple[Any,...], next_node: Any, prob: float):
        "Add entries to rules. (load_... methods must not manipulate the rules directly.)"
        assert type(start)==tuple, 'argument "start" expected a tuple, but got %r' % start
        self._max_rule_key_length = max(len(start), self._max_rule_key_length)
        #self._frozen = False
        self.rules[start][next_node] = prob
    
    def load_BuildHON_rules(self, filename: str, freeze: bool = True):
        "Load output of BuildHON+"
        with open(filename, 'r') as f:
            for line in f:
                items = line.split(' ')
                assert (len(items)>=3) and (items[-3]=='=>'), 'invalid line "%s"' % line
                key = tuple(map(int,items[:-3]))
                self.add_rule(key, int(items[-2]), float(items[-1]))
        print('%d rules read' % sum([len(v) for v in self.rules.values()]))
        if freeze:
            self.freeze_rules()

    def load_edge_list(self, edges : Iterator[Tuple[Any,Any]], directed=True):
        def add_weights(edges_: Iterator[Tuple[Any,Any]]) -> Iterator[Tuple[Any,Any,Any]]:
                for u,v in edges_:
                    yield u,v,1 # add weight
        self.load_weighted_edge_list(add_weights(edges), directed)
        
    def load_weighted_edge_list(self, edges : Iterator[Tuple[Any,Any,Any]], directed=True):
        if not directed:
            def to_directed(edges_):
                for u,v,w in edges_:
                    yield u,v,w
                    yield v,u,w
            edges = to_directed(edges)
        edges_list = list(edges) # iterating twice over edges does not work
        out_weights = defaultdict(float)
        for u,_,w in edges_list:
            out_weights[u] += w
        for u,v,w in edges_list:
            self.add_rule((u,), v, w/out_weights[u])

    @property
    def rule_keys(self) -> Iterator[Tuple[Any,...]]:
        return self.rules.keys()

    def find_rule_key(self, start: Tuple[Any,...]) -> Tuple[Any,...]:
        if start == ():
            return start
        if (len(start) > self._max_rule_key_length): # and (self._max_rule_key_length > 0):
            start = start[-self._max_rule_key_length:]
        while len(start)>0:
            if start in self.rules:
                return start
            else:
                start = start[1:]
        # issue a warning before returning an empty tuple (which correponds to the statiomary distribution)
        warnings.warn('find_rule_key did not find a key')
        return tuple()
    
    def transition_probs(self, start: Tuple[Any,...]) -> Iterator[Tuple[Tuple[Any,...],Any,float]]:
        rule_key = self.find_rule_key(start)
        probs = self.rules[rule_key]
        for next_node,prob in probs.items():
            yield(rule_key, next_node, prob)

class Node2vec_HigherOrderPathGenerator(HigherOrderPathGenerator):
    """
    Adjusts the transition probabilities for the search bias of Node2vec.
    Note, that Node2vec corresponds to DeepWalk with biased random walks.
    """
    def __init__(self, gen: ABCHigherOrderPathGenerator, p: float = 1, q: float = 1, id_format: Optional[str] = None):
        # example: id_format = 'Node2vec(p={0}, q={1})'
        super().__init__(gen._node_sort_key, id=gen._id, config=gen._config, create_EmbeddingData=gen._create_EmbeddingData)
        self._gen = gen
        self._source_path_metadata = dict(gen._source_path_metadata)
        self._target_node_metadata = dict(gen._target_node_metadata)
        if hasattr(gen, 'creator'): # see Lattice2D_2nd_order_dynamic & required for Lattice2D_EmbeddingView
            self.creator = gen.creator
        self._id_format = id_format # if id_format is provided, _id will be overwritten
        self.set_params(p, q)

    def set_params(self, p: float = 1, q: float = 1):
        "Sets the parameters for the search bias"
        self._p = p
        self._q = q
        if self._id_format is not None:
            self._id = self._id_format.format(p,q)
        self.clear_rules()
        rules_tmp = defaultdict(list)
        # copy all rules
        for key in self._gen.rule_keys:
            rules_tmp[key] = list( (next_node, prob) for _, next_node, prob in self._gen.transition_probs(key) )
        # generate rules for all direct neighbors if they do not yet exist
        direct_neighbors = set()
        for key1 in self._gen.source_paths_len1: # key1 is a tuple containing one node
            for _, key2, _ in self._gen.transition_probs(key1): # key2 is a node
                key = (key1[0], key2)
                direct_neighbors.add(key)
                if key not in rules_tmp:
                    rules_tmp[key] = list( (next_node, prob) for _, next_node, prob in self._gen.transition_probs((key2,)) )
        # add modified rules
        for key, probs in rules_tmp.items():
            if len(key) < 2:
                for next_node, prob in probs:
                    self.add_rule(key, next_node, prob)
            else:
                prev_node = key[-2]
                sum_probs = 0
                new_probs = list()
                for next_node, prob in probs:
                    # search bias
                    if next_node == prev_node:
                        prob /= p
                    elif (prev_node, next_node) not in direct_neighbors:
                        prob /= q
                    new_probs.append((next_node, prob))
                    sum_probs += prob
                for next_node, prob in new_probs:
                    self.add_rule(key, next_node, prob / sum_probs)
        self.freeze_rules()

    @property
    def config(self):
        "get configuration"
        cfg=dict(init_class=self.__class__.__name__, init_id=self._id, p=self._p, q=self._q)
        cfg.update(self._config)
        return cfg

class CrossValidation_HigherOrderPathGenerator(ABCHigherOrderPathGenerator):
    """
    Adjusts the transition probabilities for cross validation
    """
    def __init__(self, gen: ABCHigherOrderPathGenerator, excluded_edges : Iterator[Tuple[Any,Any]], id: Optional[str] = None):
        self._gen = gen
        self._excluded_edges = set(excluded_edges)
        super().__init__(gen._node_sort_key, id, gen._config, create_EmbeddingData=gen._create_EmbeddingData)
        self._source_path_metadata = dict(gen._source_path_metadata)
        self._target_node_metadata = dict(gen._target_node_metadata)
        # init rules
        self.rules = dict()
        for start in gen.rule_keys:
            if len(start)==0:
                # todo: the stationary distribution must also be adjusted
                self.rules[start]= { next_node: prob for _,next_node,prob in gen.transition_probs(start)}
                continue
            probs = list((next_node,prob) for _,next_node,prob in gen.transition_probs(start))
            d = len(probs) # out degree
            excluded = { next_node for next_node,_ in probs if (start[-1],next_node) in self._excluded_edges }
            excluded_prob = sum(prob for next_node,prob in probs if next_node in excluded)
            self.rules[start] = { next_node: 1/d if next_node in excluded else prob * (1-len(excluded)/d) / (1-excluded_prob)
                    for next_node,prob in probs}
        self.freeze_rules()

    @property
    def rule_keys(self) -> Iterator[Tuple[Any,...]]:
        return self._gen.rule_keys

    def find_rule_key(self, start: Tuple[Any,...]) -> Tuple[Any,...]:
        return self._gen.find_rule_key(start)

    @property
    def max_rule_key_length(self) -> int:
        return self._gen.max_rule_key_length

    def transition_probs(self, start: Tuple[Any,...]) -> Iterator[Tuple[Tuple[Any,...],Any,float]]:
        rule_key = self._gen.find_rule_key(start)
        probs = self.rules[rule_key]
        for next_node,prob in probs.items():
            yield(rule_key, next_node, prob)

    def write_config(self, file_object, comment: str = '', prefix: str = '', sep: str ='\t'):
        self._gen.write_config(file_object=file_object, comment=comment, prefix=prefix, sep=sep)
        file_object.write(prefix + '\n' + prefix + 'CrossValidation-HigherOrderPathGenerator:\n')
        for k,v in self.config.items():
            file_object.write(prefix + '%s%s%s\n' % (k,sep,v))

class Embedding_HigherOrderPathGenerator(HigherOrderPathGenerator):
    "Generator based on the decode-method of an asymmetric embedding"
    def __init__(self, emb: 'ABCAsymmetricEmbedding', use_neighborhood: bool = True, no_self_loops: bool = False, id: Optional[str] = None, **kwargs):
        config = dict(init_use_neighborhood=use_neighborhood, init_no_self_loops=no_self_loops, init_emb=emb._id, **kwargs)
        super().__init__(emb._gen._node_sort_key, id=id, config=config, create_EmbeddingData=emb._gen._create_EmbeddingData)
        gen = emb._gen
        self._emb = emb
        self._source_path_metadata = dict(gen._source_path_metadata)
        self._target_node_metadata = dict(gen._target_node_metadata)
        if hasattr(gen, 'creator'): # see Lattice2D_2nd_order_dynamic & required for Lattice2D_EmbeddingView
            self.creator = gen.creator
        str2nodes = { key_str: key for key,key_str in zip(emb.target.keys,emb.target.keys_str) }
        for start in emb.source.keys:
            for next_node_str,prob in emb.decode_path(start=start, use_neighborhood=use_neighborhood, no_self_loops=no_self_loops, normalize=True, **kwargs).items():
                if prob>0:
                    self.add_rule(start, str2nodes[next_node_str], prob)
        # todo: calculate stationary distribution from the first-order rules
        for _,next_node,prob in gen.transition_probs(start=()): # copy stationary distribution from original generator
            self.add_rule((), next_node, prob)
        self.freeze_rules(check=False)

    def write_config(self, file_object, comment: str = '', prefix: str = '', sep: str ='\t'):
        self._emb.write_config(file_object=file_object, comment=comment, prefix=prefix, sep=sep)
        file_object.write(prefix + '\n' + prefix + 'Embedding-HigherOrderPathGenerator:\n')
        for k,v in self.config.items():
            file_object.write(prefix + '%s%s%s\n' % (k,sep,v))
