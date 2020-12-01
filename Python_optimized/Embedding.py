from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, List, Tuple, Optional, Union
import warnings
import numpy as np
import pandas as pd
import math
import collections
import os

from HigherOrderPathGenerator import ABCHigherOrderPathGenerator, HigherOrderPathGenerator, Node2vec_HigherOrderPathGenerator
from gensim.models import Word2Vec
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from zlib import adler32, crc32
import sys

# While the public interface of the embedding classes presents the 
# embedding matrices as pandas DataFrames, they are internally stored 
# in numpy ndarrays together with dictionaries for the row and column 
# names. Besides better performance, the representation of paths as 
# tuples did not fit well with pandas - where the paths had to be converted 
# into strings. (And converting paths into str or int is basically the 
# same in terms of readibility of the code.)

class ABCEmbedding(ABC):
    def __init__(self, gen: ABCHigherOrderPathGenerator, dimension: int, symmetric:bool=True):
        self._gen = gen
        self._symmetric = symmetric
        self._dimension = dimension
        self._id = type(self).__name__

    @property
    @abstractmethod
    def dimension(self):
        "effective dimension of embedding"
        pass

    # @abstractmethod
    # def node2key(self, node: Any, use_source: bool=False):
    #     pass

    # @abstractmethod
    # def __getitem__(self, key) -> pd.Series:
    #     "Returns a row of the embedding matrix."
    #     pass

    @abstractmethod
    def decode_path(self, start: Tuple[Any,...], **kwargs) -> pd.Series:
        pass

    @property
    @abstractmethod
    def config(self):
        "get configuration"
        pass

    def write_config(self, file_object, comment: str = '', prefix: str = '', sep: str ='\t'):
        self._gen.write_config(file_object=file_object, comment=comment, prefix=prefix, sep=sep)
        file_object.write(prefix + '\n' + prefix + 'Embedding:\n')
        for k,v in self.config.items():
            file_object.write(prefix + '%s%s%s\n' % (k,sep,v))

class ABCSymmetricEmbedding(ABCEmbedding):
    def __init__(self, gen: ABCHigherOrderPathGenerator, dimension: int):
        super().__init__(gen, dimension, symmetric=True)
        self._data = gen.create_target_embedding_data(self, dimension)

    ## for convenience use the same property names (source and target) as in ABCAsymmetricEmbedding for the embedding data
    @property
    def source(self):
        "Embedding data (source and target are identical)"
        return self._data

    @property
    def target(self):
        "Embedding data (source and target are identical)"
        return self._data

    # def __getitem__(self, key) -> pd.Series:
    #     idx = self._data._keys_dict[key]
    #     return pd.Series(self._data._embedding[idx,:], name=self.key2str(key))

    # def node2key(self, node: Any, use_source: bool=False):
    #     return node

    @property
    def key2str(self):
        return self._data.key2str

    @key2str.setter
    def key2str(self, key2str):
        self._data.key2str = key2str

    # @property
    # def nodes(self) -> Iterator[Any]:
    #     return self._data.keys

    # @property
    # def nodes_str(self) -> List[str]:
    #     return self._data.keys_str

    # @property
    # def embedding(self) -> pd.DataFrame:
    #     return self._data.embedding

    @property
    def dimension(self):
        "effective dimension of embedding"
        return self._data._dimension

    def _decode_raw(self, node: Any, step = None) -> np.array:
        """Calculate the transition probabilities for start from the embedding.
        Default implementation assumes the Skip-gram model.
        (The parameter step is not used.)
        """
        return np.exp(self._data._embedding[self._data._keys_dict[node]] @ self._data._embedding.T)

    def decode(self, node: Any, use_neighborhood: bool = False, no_self_loops: bool = False, normalize: bool = True, **kwargs) -> pd.Series:
        """Calculate probabilities from the embedding.
        The parameter node must contain a value from self.nodes.

        Parameters
        ----------

        node : Any

            key of the embedding.
        
        use_neighborhood : bool, optional (default = False)

            restrict the probabilities to the neighborhood (using knowledge about network toology).
        
        no_self_loops : bool, optional (default = False)

            set the probability of a self loop to zero.

        normalize : bool, optional (default = True)

            normalize the probabilities, such that they sum up to one.
        
        **kwargs are passed to _decode_raw(node)
        """
        data = self._decode_raw(node=node, **kwargs)
        if use_neighborhood:
            data_n = np.zeros(len(data))
            for _,next_node,_ in self._gen.transition_probs(start=(node,)):
                i = self._data._keys_dict[next_node]
                data_n[i] = data[i]
            data = data_n
        if no_self_loops:
            data[self._data._keys_dict[node]] = 0
        if normalize:
            data = data / max(1e-16, data.sum())
        return pd.Series(data, index=self._data.keys_str, name=self.key2str(node))

    def decode_path(self, start: Tuple[Any,...], **kwargs) -> pd.Series:
        "Returns decode() for the last node of the path."
        return self.decode(start[-1], **kwargs)

class ABCAsymmetricEmbedding(ABCEmbedding):
    def __init__(self, gen: ABCHigherOrderPathGenerator, dimension: int, pairwise: bool):
        super().__init__(gen, dimension, symmetric=False)
        self._source = gen.create_source_embedding_data(self, dimension, pairwise)
        self._target = gen.create_target_embedding_data(self, dimension)

    @property
    def source(self):
        "Source embedding data"
        return self._source

    @property
    def target(self):
        "Target embedding data"
        return self._target

    # def __getitem__(self, key) -> pd.Series:
    #     """
    #     If the key contains a node, a row of the target_embedding is returned.
    #     And if the key contains a tuple of nodes, a row of the source_embedding matrix is returned instead.
    #     """
    #     idx = self._target._keys_dict.get(key, None)
    #     if idx is not None:
    #         return pd.Series(self._target._embedding[idx,:], name=self.node2str(key))
    #     idx = self._source._keys_dict.get(key, None)
    #     if idx is not None:
    #         return pd.Series(self._source._embedding[idx,:], name=self.path2str(key))
    #     else:
    #         raise IndexError('Unknown key %s' % key, key=key)

    # def node2key(self, node: Any, use_source: bool=False):
    #     return (node,) if use_source else node

    @property
    def path2str(self):
        return self._source.key2str

    @path2str.setter
    def path2str(self, key2str):
        self._source.key2str = key2str

    @property
    def node2str(self):
        return self._target.key2str

    @node2str.setter
    def node2str(self, key2str):
        self._target.key2str = key2str

    # @property
    # def source_paths(self) -> Iterator[Tuple[Any,...]]:
    #     return self._source.keys
    
    # @property
    # def source_paths_str(self) -> List[str]:
    #     return self._source.keys_str

    # @property
    # def target_nodes(self) -> Iterator[Any]:
    #     return self._target.keys

    # @property
    # def target_nodes_str(self) -> List[str]:
    #     return self._target.keys_str

    # @property
    # def source_embedding(self) -> pd.DataFrame:
    #     return self._source.embedding

    # @property
    # def target_embedding(self) -> pd.DataFrame:
    #     return self._target.embedding
    
    @property
    def dimension(self):
        "effective dimension of embedding"
        return self._source._embedding.shape[1]

    def _decode_raw(self, start: Tuple[Any,...], step = None) -> np.array:
        """Calculate the transition probabilities for start from the embedding.
        Default implementation assumes the Skip-gram model.
        (The parameter step is not used.)
        """
        return np.exp(self._source._embedding[self._source._keys_dict[start]] @ self._target._embedding.T)

    def decode(self, start: Tuple[Any,...], use_neighborhood: bool = False, no_self_loops: bool = False, normalize: bool = True, **kwargs) -> pd.Series:
        """Calculate probabilities from the embedding.
        The parameter start must contain a value from self.source.keys.

        Parameters
        ----------

        start : Tuple[Any, ...]

            key of the source embedding.
        
        use_neighborhood : bool, optional (default = False)

            restrict the probabilities to the neighborhood (using knowledge about the first-order network topology).
        
        no_self_loops : bool, optional (default = False)

            set the probability of a self loop to zero.

        normalize : bool, optional (default = True)

            normalize the probabilities, such that they sum up to one.
        
        **kwargs are passed to _decode_raw(start)
        """
        data = self._decode_raw(start=start, **kwargs)
        if use_neighborhood:
            data_n = np.zeros(len(data))
            for _,next_node,_ in self._gen.transition_probs(start=start[-1:]):
                i = self._target._keys_dict[next_node]
                data_n[i] = data[i]
            data = data_n
        if no_self_loops:
            data[self._target._keys_dict[start[-1]]] = 0
        if normalize:
            data = data / max(1e-16, data.sum())
        return pd.Series(data, index=self._target._keys_str, name=self.path2str(start))

    def decode_path(self, start: Tuple[Any,...], **kwargs) -> pd.Series:
        "Finds the matching source_path and returns decode()"
        if start == ():
            return None
        if start in self._source._keys_dict:
            return self.decode(start, **kwargs)
        if (len(start) > self._gen.max_rule_key_length): # and (self._gen.max_rule_key_length > 0):
            start = start[-self._gen.max_rule_key_length:]
        while len(start)>0:
            if start in self._source._keys_dict:
                return self.decode(start, **kwargs)
            else:
                start = start[1:]
        return None

    @staticmethod
    def factor_matrix(mat: np.ndarray, dimension: int=None, check=False)-> Tuple[np.ndarray, np.ndarray]:
        "Factor matrix using (truncated) SVD"
        if (dimension is None) or (dimension >= min(mat.shape)): # SVD
            dimension = min(mat.shape)
            U,S,Vh = svd(mat)
            sqrtS = S**0.5
            if mat.shape[0] > mat.shape[1]:
                U = U[:, :dimension:] # U[:,:1] drops 2nd dimension (squeezing), while U[:, :1:] does not
            elif mat.shape[0] < mat.shape[1]:
                Vh = Vh[:dimension:, :]
            source_embedding = U * (sqrtS[np.newaxis,:])
            target_embedding = np.transpose((sqrtS[:,np.newaxis]) * Vh)
        else:
            tsvd = TruncatedSVD(n_components=dimension, random_state=1)
            tsvd.fit(mat) # returns svd
            W = tsvd.fit_transform(mat) # = U * Sigma, shape=(mat.shape[0], dimension)
            H = tsvd.components_ # = V', shape=(dimension, mat.shape[1])
            # checking factorization
            #print('diff %g' % abs(mat - W@H).max())
            S = np.linalg.norm(W,axis=0) # len(S) == dimension
            S = S.clip(min=1e-20) # avoid division by zero
            source_embedding = W / ((S**0.5)[np.newaxis,:]) # = W @ np.diag(S**-0.5)
            target_embedding = np.transpose(H * ((S**0.5)[:,np.newaxis])) # = (np.diag(S**0.5) @ H).T
        if check: # checking factorization
            error = np.linalg.norm(mat - source_embedding @ target_embedding.transpose())
            print('Approximation error %g' % (error/np.linalg.norm(mat)))
        return (source_embedding, target_embedding)

class Generic_SkipGram_Embedding(ABCSymmetricEmbedding):
    "Wrapper for embeddings based on the skip-gram model calculated externally (e.g., LINE, Node2vec)"
    def __init__(self, gen: ABCHigherOrderPathGenerator, emb_path : str, id: str = None, parse_node=lambda x:x, binary: bool=False, config=dict()):
        assert binary==False, 'not implemented'
        with open(emb_path, 'r') as f:
            dims = f.readline().split(' ')
            emb = pd.read_csv(f, sep=' ', header=None, comment='%', index_col=0)
        if all(emb[emb.columns[-1]].isna()):
            # The output of LINE has a training space resulting in a columns with NaN; Node2vec does not.
            emb.drop(emb.columns[-1], axis=1, inplace=True)
        super().__init__(gen, dimension=int(dims[1]))
        self._id = os.path.splitext(os.path.split(emb_path)[-1])[0] if id is None else id
        self._emb_path = emb_path
        #self._binary = binary
        assert self._data._embedding.shape == emb.values.shape, f'shapes do not match: {self._data._embedding.shape}!={emb.values.shape}'
        # verify row names and sort rows
        emb.index = emb.index.map(parse_node)
        keys_emb = set(emb.index)
        keys_gen = set(gen.target_nodes)
        if len(keys_emb.symmetric_difference(keys_gen))>0:
            print('keys do not match (did you specify parse_node?)')
            print('only in embedding:', keys_emb-keys_gen)
            print('only in generator:', keys_gen-keys_emb)
            assert False
        ##emb.sort_index(axis=0, inplace=True, key=gen._node_sort_key) # requires pandas 1.1 and _node_sort_key must be vectorized
        emb_sort = pd.Series(data=emb.index, index=emb.index).map(gen._node_sort_key).sort_values().index
        emb = emb.loc[emb_sort]
        self._data._embedding = emb.values
        self._config = config
        
    def train(self):
        pass

    @property
    def config(self):
        "get configuration"
        cfg = dict(init_class=self.__class__.__name__, init_gen=self._gen._id, init_emb_path = self._emb_path, init_id=self._id)#, init_binary=self._binary)
        cfg.update(self._config)
        return cfg

class HON_DeepWalk_Embedding(ABCSymmetricEmbedding):
    """
    Adapts [DeepWalk] to random walks in higher order models, see
    [DeepWalk] Perozzi B., Al-Rfou R., and Skiena S. (2014)
    'Deepwalk: Online learning of social representations', 
    https://doi.org/10.1145/2623330.2623732

    Uses gensim.models.Word2Vec for the embedding, which treats the random 
    walks as bidirectional. Hence, the embedding is symmetric.
    """
    def __init__(self, gen: ABCHigherOrderPathGenerator, dimension: int, reuse_walks: bool = False):
        super().__init__(gen, dimension)
        self._id = 'DeepWalk'
        self._reuse_walks = reuse_walks
        self._walks = None
        self._training_config= {'num_walks': 0, 'walk_length': 0, 'random_seed': 0}

    # https://stackoverflow.com/questions/34831551
    # Word2vec uses the hash function to initialize the vector for each word, no need for a cryptographic hash function
    if sys.hash_info.width == 64:
        @staticmethod
        def str_hash(data:str) -> int:
            "deterministic hash function for Word2Vec (64 bit)"
            b = data.encode()
            return adler32(b) + (crc32(b) << 32)
    else:
        @staticmethod
        def str_hash(data:str) -> int:
            "deterministic hash function for Word2Vec (32 bit)"
            return crc32(data.encode())

    def train(self, num_walks:int=100, walk_length:int=80, window_size=10, num_iter:int=1, min_count:int=0,
            hs:bool = True, negative:int=5, workers:int=1, random_seed=None, replace_hash:bool=True, use_numba: bool = True, **kwargs):
        # reuse_walks allows for efficiently evaluating different params 
        # of Word2vec (e.g., negative, num_iter).
        # walks depend only on num_walks, walk_length, random_seed, and use_numba.
        # (Although, use_numba should have no effect - besides speed.)
        tc = self._training_config # config of previous training
        can_reuse_walks = tc['num_walks']==num_walks and tc['walk_length']==walk_length and tc['random_seed']==random_seed and tc['use_numba']==use_numba
        self._training_config = dict(num_walks=num_walks, walk_length=walk_length, window_size=window_size, 
                num_iter=num_iter, min_count=min_count, hs=hs, negative=negative, workers=workers, 
                random_seed=random_seed, replace_hash=replace_hash, use_numba=use_numba, **kwargs)
        if (not can_reuse_walks) or (self._walks is None):
            walks = list()
            rng = np.random.default_rng(random_seed) # for shuffle and random walks
            for _ in range(num_walks):
                nodes_s = list(self._data._keys)
                rng.shuffle(nodes_s)
                if use_numba:
                    start_list = list((node,) for node in nodes_s)
                    walks.extend(self._gen.random_walks(start_list, num_steps=walk_length, rng=rng, include_start=True))
                else:
                    for node in nodes_s:
                        walks.append(self._gen.random_walk(start=(node,), num_steps=walk_length, rng=rng))
            walks = [list(map(str, walk)) for walk in walks]
            if self._reuse_walks:
                self._walks = walks
        else:
            walks = self._walks
        if replace_hash:
            if 'hashfxn' in kwargs:
                print('The parameter hashfxn is ignored because of replace_hash=True')
            kwargs['hashfxn'] = self.str_hash
        model = Word2Vec(walks, size=self._dimension, window=window_size, sg=1, min_count=min_count, iter=num_iter,
            hs=1 if hs else 0, negative=0 if hs else negative, workers=workers, seed=random_seed, **kwargs)
        self.model = model # debug
        for iv,v in enumerate(self._data.keys):
            self._data._embedding[iv,:] = model.wv[str(v)]

    @property
    def config(self):
        "get configuration"
        cfg = dict(init_class=self.__class__.__name__, init_gen=self._gen._id, init_dimension=self._dimension, init_id=self._id)
        cfg.update(self._training_config)
        return cfg

class HON_Node2vec_Embedding(HON_DeepWalk_Embedding):
    """
    Adapts [Node2vec] to random walks in higher order models, see        
    [Node2vec] Grover A. and Leskovec J. (2016)
    'node2vec: Scalable Feature Learning for Networks', https://doi.org/10.1145/2939672.2939754

    The differences between Node2vec and DeepWalk are:
    - Node2vec uses biased random walks, implemented in Node2vec_HigherOrderPathGenerator
    - Both rely on gensim.models.Word2Vec, but Node2vec uses negative sampling and DeepWalk uses hierarchical softmax.
    """
    def __init__(self, gen: ABCHigherOrderPathGenerator, dimension: int = 128, p: float = 1, q: float = 1, reuse_walks:bool=False):
        gen_n2v = Node2vec_HigherOrderPathGenerator(gen, p, q)
        super().__init__(gen_n2v, dimension, reuse_walks)
        self._id = 'Node2vec(p={0}, q={1})'.format(p,q)

    def set_params(self, p: float = 1, q: float = 1):
        self._gen.set_params(p, q) # do not call this directly
        self._id = 'Node2vec(p={0}, q={1})'.format(p,q)
        self._walks = None
        #self._data._embedding = np.zeros(self._data._embedding.shape) # reset embedding

    def train(self, negative:int=5, hs:bool = False, **kwargs): # Node2Vec uses negative sampling by default
        super().train(hs=hs, negative=negative, **kwargs)

class HONEM_Embedding(ABCAsymmetricEmbedding):
    """
    Calculates the embedding according to
    [HONEM] Saebi M., Ciampaglia G., Kaplan L., and Chawla N. (2019) 
    'HONEM: Network Embedding Using Higher-Order Patterns in Sequential Data', arXiv:1908.05387

    The instance of class HigherOrderPathGenerator is assumed to contain the rules detected by BuildHON+, see
    [BuildHON+] Xu J., Saebi M., Ribeiro B., Kaplan L., and Chawla N. (2017)
    'Detecting Anomalies in Sequential Data with Higher-order Networks', arXiv:1712.09658
    """
    def neighborhood_matrix(self, order: int=1, sort=True) -> pd.DataFrame:
        "Calculates the 'v-th order neighborhood matrix' defined in the [HONEM] paper"
        keys = [key for key in self._gen.source_paths if len(key)==order]
        data = [(self.path2str((key[0],)), self.node2str(next_node), prob) 
                for key in keys for _, next_node, prob in self._gen.transition_probs(key)]
        data_df = pd.DataFrame(data, columns=['src','trg','prob'])
        res = data_df.groupby(['src','trg']).mean().unstack(fill_value=0)
        res.columns = [c[1] for c in res.columns] # c[0] == 'prob'
 
        # ensure that neighborhood matrices of different orders have identical indices and columns
        all_source_nodes = { self.path2str((key[0],)) for key in self._gen.source_paths_len1 }
        for v in all_source_nodes.difference(res.index):
            res.loc[v]=0
        all_target_nodes = { self.node2str(node) for node in self._gen.target_nodes }
        for v in all_target_nodes.difference(res.columns):
            res[v]=0

        if sort:
            res = res.loc[self._source._keys_str]
            res = res[self._target._keys_str]
        return res

    def __init__(self, gen: HigherOrderPathGenerator, dimension: int):
        super().__init__(gen, dimension, pairwise=True)
        self._id = 'HONEM'
    
    def train(self):
        # calculate neighborhood matrix
        neighborhood = self.neighborhood_matrix(1)
        if self._gen.max_rule_key_length > 1:
            for order in range(2, self._gen.max_rule_key_length+1):
                neighborhood = neighborhood + math.exp(1-order) * self.neighborhood_matrix(order, sort=False)
        # sort
        neighborhood = neighborhood.loc[self._source._keys_str]
        neighborhood = neighborhood[self._target._keys_str]

        source_embedding, target_embedding = self.factor_matrix(neighborhood.values, self._dimension)
        self._source._embedding = source_embedding
        self._target._embedding = target_embedding

    def _decode_raw(self, start: Tuple[Any,...], step = None) -> np.array:
        "Calculate the transition probabilities for start from the embedding. (HONEM does not use the skip-gram model)"
        assert type(start) is tuple and len(start)==1, 'start must be a tuple of length 1'
        return self._source._embedding[self._source._keys_dict[start]] @ self._target._embedding.T
    
    @property
    def config(self):
        "get configuration"
        return dict(init_class=self.__class__.__name__, init_gen=self._gen._id, init_dimension=self._dimension, init_id=self._id)

class HON_NetMF_Embedding(ABCAsymmetricEmbedding):
    """
    Adapts [NetMF] to random walks in a higher-order model. (Additionally, the assumption of undirected graphs is dropped.)

    [NetMF] Qiu, Jiezhong and Dong, Yuxiao and Ma, Hao and Li, Jian and Wang, Kuansan and Tang, Jie (2018)
    'Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and Node2vec', https://doi.org/10.1145/3159652.3159706
    """
    def __init__(self, gen: ABCHigherOrderPathGenerator, dimension: int, pairwise=False):
        super().__init__(gen, dimension, pairwise)
        self._id = 'NetMF_pairs' if pairwise else 'NetMF_paths'
        self._pairwise = pairwise
        self._training_config= {}
        # caching the PMI calculation
        self._window_size = 0
        self._PMI = None

    def train(self, window_size: int, negative: int = 1, optimized: bool = True, use_numba : bool = True):
        assert window_size > 0, 'window_size must be a positive integer'
        self._training_config = dict(window_size=window_size, negative=negative, optimized=optimized, use_numba=use_numba)
        if window_size != self._window_size:
            # calculate pointwise mutual information (PMI) - this takes some time
            if use_numba and optimized:
                PMI = self._gen.walk_probs(num_steps=window_size, pairwise=self._pairwise, aggregate_steps=True, step_factor=1.0)
                #sd = np.zeros(len(self._gen.target_nodes)) # stationary distribution (for target nodes)
                sd = np.zeros(len(self._target._keys)) # stationary distribution (for target nodes)
                for _,v,p in self._gen.transition_probs(start=()):
                    #sd[self._gen._target_nodes_dict[v]] = p
                    sd[self._target._keys_dict[v]] = p
                PMI /= sd[np.newaxis,:]
            else:
                PMI = np.zeros(shape=(len(self._source._keys), len(self._target._keys)))
                idx = { v:i for i,v in enumerate(self._target._keys) }
                # stationary distribution (for target nodes)
                sd = { v:p for _,v,p in self._gen.transition_probs(start=()) }
                if optimized:
                    # We want to enumerate all paths and aggregate the visiting 
                    # probabilities of the indiviidual nodes. However, enumerating the 
                    # paths in a depth-first style is generally expensive (exponential
                    # growth in the length of number of paths). Instead use a breadth-
                    # first enumeration of the paths. Because of the higher-order Markov 
                    # property of the transition probabilites, we need only to keep track
                    # of the last k=max_rule_key_length nodes visited. This avoids the 
                    # exponential growth. 
                    # There are max. (#nodes ** max_rule_key_length) different subpaths.
                    for iu,u in enumerate(self._source._keys): # self._source._keys_dict contains additionally the key ()
                        paths = {u: 1/window_size}
                        for _ in range(window_size):
                            paths_new = collections.defaultdict(float)
                            for source,source_prob in paths.items():
                                for _,next_node,prob in self._gen.transition_probs(source):
                                    prob_new = source_prob * prob
                                    PMI[iu,idx[next_node]] += prob_new / sd[next_node]
                                    source_new = self._gen.find_rule_key((*source, next_node))
                                    paths_new[source_new] += prob_new
                            paths = paths_new
                else:
                    # slower implementation enumerating all paths (debugging only)
                    for iu,u in enumerate(self._source._keys):
                        for path,prob in self._gen.path_probs(u, num_steps = window_size):
                            prob_div = prob / window_size # = probability of this path conditional to start=u, divided by window_size
                            for v in path[len(u):]: # skip the start from the path
                                iv = idx[v]
                                PMI[iu,iv] += prob_div / sd[v]
            # cache calculation
            self._window_size = window_size
            self._PMI = PMI
        self._negative = negative
        M = self._PMI / negative
        mat = np.log(M.clip(1))
        source_embedding, target_embedding = self.factor_matrix(mat, self._dimension)
        self._source._embedding = source_embedding
        self._target._embedding = target_embedding

    @property
    def PMI(self) -> pd.DataFrame:
        return pd.DataFrame(self._PMI, index=self._source._keys_str, columns=self._target._keys_str)
    
    @property
    def config(self):
        "get configuration"
        cfg = dict(init_class=self.__class__.__name__, init_gen=self._gen._id, init_dimension=self._dimension, init_id=self._id, init_pairwise=self._pairwise)
        cfg.update(self._training_config)
        return cfg

class HON_GraRep_Embedding(ABCAsymmetricEmbedding):
    """
    Adapts [GraRep] to random walks in a higher-order model.

    [GraRep] Cao, Shaosheng and Lu, Wei and Xu, Qiongkai (2015)
    'GraRep: Learning Graph Representations with Global Structural Information', https://doi.org/10.1145/2806416.2806512
    """
    def __init__(self, gen: ABCHigherOrderPathGenerator, dimension: int, num_steps: int = 4, pairwise:bool=True, neg_stationary:bool=False):
        """
        Init

        Parameters
        ----------
        dimension : int, >0
            The dimension of the embedding for each individual step.
            Effectively, the dimension is num_steps * dimension.
        
        num_steps : int, >0 (default = 4)
            ..
        
        pairwise : bool (default = True)
            Indicates whether nodes (True) or paths are embedded.
        
        neg_stationary : bool (default = False)
            The PMI is divided by the corresponding averages (False, as in GraRep) 
            or by the stationary distribution (True, as in HON NetMF).
        """
        self._dimension_per_step = dimension = min(dimension, len(gen.target_nodes)) # dimension returned by factor_matrix()
        super().__init__(gen, self._dimension_per_step * num_steps, pairwise)
        self._dimension = dimension # specified dimension
        self._id = 'GraRep_pairs' if pairwise else 'GraRep_paths'
        self._pairwise = pairwise
        self._num_steps = num_steps
        self._neg_stationary = neg_stationary # distribution for negative samples: stationary (True) or averages (False)
        self._training_config= {}
        # caching the PMI calculation
        self._PMI = None

    def train(self, negative: int = 1, normalize: bool = False, use_numba: bool = True):
        # optimized PMI calculation from HON_NetMF_Embedding (with small modifications).
        # HON_NetMF_Embedding._PMI matches self._PMI.mean(axis=2) if window_size equals num_steps and neg_stationary=True
        self._training_config = dict(negative=negative, normalize=normalize, use_numba=use_numba)
        if self._PMI is None:
            idx = { v:i for i,v in enumerate(self._target._keys) }
            if use_numba:
                PMI = self._gen.walk_probs(num_steps=self._num_steps, pairwise=self._pairwise, aggregate_steps=False)
            else:
                PMI = np.zeros(shape=(len(self._source._keys), len(self._target._keys_str), self._num_steps))
                for iu,u in enumerate(self._source._keys):
                    paths = {u: 1} # 1 instead of 1/sindow_size in NetMF
                    for step0 in range(self._num_steps): # step0 = step - 1
                        paths_new = collections.defaultdict(float)
                        for source,source_prob in paths.items():
                            for _,next_node,prob in self._gen.transition_probs(source):
                                prob_new = source_prob * prob
                                PMI[iu, idx[next_node], step0] += prob_new
                                source_new = self._gen.find_rule_key((*source, next_node))
                                paths_new[source_new] += prob_new
                        paths = paths_new
            # The PMI values calculated above correspond to p(c|w), and we stil 
            # have to divide it by p(c) - or similar. GraRep divides it by 
            # sum_w p(c|w) / |V|, see Table 1 in [GraRep]. (The division by the 
            # number of nodes |V| is borrowed from the parameter beta.)
            # neg_stationary=True instead divides by the stationary distribution 
            # for comparison with NetMF.
            averages = np.zeros(PMI.shape[1])
            if self._neg_stationary:
                for _,v,p in self._gen.transition_probs(start=()): # stationary distribution
                    averages[idx[v]] += p

            for step0 in range(self._num_steps): # step0 = step - 1
                if not self._neg_stationary:
                    averages = PMI[:,:,step0].mean(axis=0) # divide by mean over column
                PMI[:,:,step0] /= averages[np.newaxis,:]
            
            self._PMI = PMI
        self._negative = negative
        for step0 in range(self._num_steps):
            M = self._PMI[:, :, step0] / negative
            mat = np.log(M.clip(1))
            source_embedding, target_embedding = self.factor_matrix(mat, self._dimension_per_step)
            indices = slice(step0 * self._dimension_per_step, (step0 + 1) * self._dimension_per_step)
            # Table 1 in [GraRep] does not mention normalizing the individual embeddings before concatenating them together.
            # However, https://github.com/ShelsonCao/GraRep scales each embedding by its L2 norm.
            if normalize:
                source_embedding = source_embedding / np.linalg.norm(source_embedding)
                target_embedding = target_embedding / np.linalg.norm(target_embedding)
            self._source._embedding[:, indices] = source_embedding
            self._target._embedding[:, indices] = target_embedding

    def PMI(self, step: int = 1) -> pd.DataFrame:
        return pd.DataFrame(self._PMI[:, :, step - 1], index=self._source._keys_str, columns=self._target._keys_str)

    def _decode_raw(self, start: Tuple[Any,...], step: int = 1) -> pd.Series:
        indices = slice((step - 1) * self._dimension_per_step, step * self._dimension_per_step)
        return np.exp(self._source._embedding[self._source._keys_dict[start], indices] @ self._target._embedding[:,indices].T)

    @property
    def config(self):
        "get configuration"
        cfg = dict(init_class=self.__class__.__name__, init_gen=self._gen._id, init_dimension=self._dimension, init_id=self._id, 
                   init_num_steps=self._num_steps, init_pairwise=self._pairwise, init_neg_stationary=self._neg_stationary)
        cfg.update(self._training_config)
        return cfg

class HON_Transition_Hierarchical_Embedding(ABCAsymmetricEmbedding):
    """
    Extends HON_NetMF_Embedding with window_size=1 and pairwise=False by adding a penalty term which ties the 
    embeddings together along the specified hierarchy.
    Due to the penalty, we cannot use singular value decomposition (SVD) and had to rely on stochastic gradient 
    descent (SGD) instead - similar to gensim.models.Word2Vec.
    We use skip-gram with negative sampling (SGNS) as optimization criterion and added a penalty proportional to
    sum( norm(u-pu)**2  for u,pu in node_hierarchy.items() ).

    The intuition for introducing the penalty stems from the observation, that the probability of a path
    (e.g. P[A->B->C]) can be calculated from the sums of probabilities of longer paths (e.g. sum_x P[x->A->B->C])
    - assuming stationarity or introducing a 'begin of path'-symbol.
    Hence, the transition probabilities for some start path correspond to averages of transition probabilities 
    for longer paths.
    While for short paths, there is enough data (i.e. high support for the estimated transition probabilities),
    longer paths may benefit from being tied towards their parent path (i.e. path[1:]).
    The notion of "short" and "longer" is controlled by the parameter min_length of calc_node_hierarchy.

    Note that, gensim.models.Word2Vec tries very hard to speed calculation up, as explained by a blog article of 
    its author. Unfortunately, this implementation will be much slower.
    """
    @staticmethod
    def calc_node_hierarchy(gen: ABCHigherOrderPathGenerator, min_length:int=1) -> Dict[Tuple[Any,...],Tuple[Any,...]]:
        """
        Calculates the default node_hierarchy, which is used in __init__(..., node_hierarchy='calc').

        min_length specifies the minimal length of those paths, which are affected by the penalty.
        """
        assert min_length > 0, 'min_length must be > 1'
        return {key: key[1:] for key in gen.rule_keys if len(key) >= min_length + 1 }

    def __init__(self, gen: ABCHigherOrderPathGenerator, dimension: int=128, 
            node_hierarchy:Union[str,Dict[Tuple[Any,...],Tuple[Any,...]]]='calc', seed=None, neg_stationary:bool=True):
        self._neg_stationary = neg_stationary # distribution for negative samples: stationary (True) or uniform (False)
        super().__init__(gen, dimension, pairwise=False)
        self._id = 'Hierarchical_GF' # todo
        self.reset(seed)
        if node_hierarchy == 'calc':
            node_hierarchy = self.calc_node_hierarchy(gen)
        assert isinstance(node_hierarchy, collections.abc.Mapping), "node_hierarchy must be either the string 'calc' or a dictionary with keys containing paths and values containing the parent of each path, i.e. key[1:]"
        self._parent = node_hierarchy
        children = collections.defaultdict(set)
        for u,pu in node_hierarchy.items():
            children[pu].add(u)
        self._children = children

    def clone(self):
        "Clones the instance"
        res = HON_Transition_Hierarchical_Embedding(self._gen, self._dimension, self._parent, None, self._neg_stationary)
        res._total_steps = self._total_steps
        res._source._embedding = self._source._embedding.copy()
        res._target._embedding = self._target._embedding.copy()
        res._rng.bit_generator.state = self._rng.bit_generator.state
        res._training_history = list(self._training_history)
        return res

    def reset(self, seed=None):
        "Resets the embedding (matrices are initialized with random values)."
        dimension = self._dimension
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._total_steps = 0
        self._source._embedding = self._rng.normal(0, 1/dimension, (len(self._source._keys),dimension))
        self._target._embedding = self._rng.normal(0, 1/dimension, (len(self._target._keys),dimension))
        self._training_history = list()

    @staticmethod
    def sigmoid(x: float) -> float:
        # https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python ... ore use lookup table
        # sigmoid(-710) raises a math range error
        return  1 / (1  + math.exp(-max(x,-700)))

    def _calc_objective(self) -> Tuple[float,float,float]:
        "calculates the loss function (SGNS and penalty are returned separately)"
        loss = 0
        loss_neg = 0
        # neg_probs = distribution of negative samples
        if self._neg_stationary: # stationary distribution
            neg_probs = { next_node:prob for _,next_node,prob in self._gen.transition_probs(start=()) }
        else: # uniform distribution
            n_target = len(self._target._keys)
            neg_probs = { next_node:1/n_target for next_node in self._target._keys }
        for iu,u in enumerate(self._source._keys): # uniform sampling
            probs = { next_node:prob for _,next_node,prob in self._gen.transition_probs(u) }
            for iv,v in enumerate(self._target._keys):
                score = self._source._embedding[iu,:] @ self._target._embedding[iv,:]
                sigmoid_value = self.sigmoid(score)
                loss -= math.log(sigmoid_value) * probs.get(v,0) 
                loss_neg -= math.log(1-sigmoid_value) * neg_probs[v]
        loss_penalty = 0
        for u,pu in self._parent.items():
            iu = self._source._keys_dict[u]
            ipu = self._source._keys_dict[pu]
            loss_penalty += np.linalg.norm(self._source._embedding[iu,:] - self._source._embedding[ipu,:])**2
        return (loss, loss_neg, loss_penalty) # objective is loss + negative * loss_neg + penalty * loss_penalty

    def _update(self, iu, iv, label, learning_rate):
        "SGD update"
        eu = self._source._embedding[iu,:]
        ev = self._target._embedding[iv,:]
        score = eu @ ev
        grad = learning_rate * (self.sigmoid(score) - label)
        self._source._embedding[iu,:] -= grad * ev
        self._target._embedding[iv,:] -= grad * eu

    def _update_hierarchy(self, u, iu, learning_rate, penalty, max_start_len: int):
        "Calculates the gradient of the penalty and updates the source embedding accordingly."
        factor = min(2 * learning_rate * penalty, 1) # avoid overshooting, which could result in a "math domain error" in the sigmoid calculation
        if u in self._children:
            if len(u) == max_start_len:
                return
            ichildren = list(self._source._keys_dict[cu] for cu in self._children[u])
            ec = self._source._embedding[ichildren,:].mean(axis=0) # average embedding for all children
            self._source._embedding[iu,:] += factor * (ec - self._source._embedding[iu,:])
        
        pu = self._parent.get(u, None)
        if not pu is None:
            ipu = self._source._keys_dict[pu]
            self._source._embedding[iu,:] += factor * (self._source._embedding[ipu,:] - self._source._embedding[iu,:])

    def train(self, steps: int=1, negative: int=1, penalty: float=0, 
            learning_rate_start: float=0.0025, learning_rate_end: Optional[float]=None,
            max_start_len: Optional[int]=None, debug_objective: Optional[int]=None):
        """
        Trains the embeddings using SGD.

        Parameters
        ----------

        steps : int, >0

            The number of SGD updates (not counting negative samples) per each path in source_paths.

        negative : int, >= 0

            Number of negative samples.

        penalty: float, >= 0

            Factor for the penalty terms.

        learning_rate_start: float, >0

            Learning rate. Linearly decreasing learning rates are specified with learning_rate_end != None.

        learning_rate_end: float, optional (default=None)

            Allows for linearly falling learning rates.

        max_start_len: int, optional (default = None)
        
            Allows for training source embeddings corresponding to short source_paths only.
            At the end, the remaining source embeddings are replaced by their corresponding ancestors.
            (max_start_len = None disables this feature.)
            Source embeddings for a path longer than max_start_len are replaced by the ones corresponding 
            to shorter ancestors (i.e. path[-max_start_len:]).
        
        debug_objective: int, optional (default = None)
        
            If not None, this method returns the objective function evaluated in periodic intervals.
            The length of the interval is specified by the parameter debug_objective.
        """
        self._training_history.append(dict(steps=steps, negative=negative, penalty=penalty, 
            learning_rate_start=learning_rate_start, learning_rate_end=learning_rate_end,
            max_start_len=max_start_len)) # skip debug_objective
        list_of_objectives = []
        list_of_objectives_pos = []
        list_of_objectives_neg = []
        list_of_objectives_penalty = []
        learning_rate_delta = 0 if learning_rate_end is None else learning_rate_end - learning_rate_start
        if max_start_len is None:
            source_paths_s = list((u,iu) for iu,u in enumerate(self._source._keys))
            max_start_len_eff = max(len(u) for u,iu in source_paths_s)
        else:
            source_paths_s = list((u,iu) for iu,u in enumerate(self._source._keys) if len(u) <= max_start_len)
            max_start_len_eff = max_start_len
        def copy_long_key_embeddings():
            if max_start_len is not None:
                # copy embeddings corresponding to keys longer than max_start_len
                for u,iu in self._source._keys_dict.items():
                    if len(u) > max_start_len:
                        pu = u[(-max_start_len_eff):] # FIXME: hardcoded assumption about hode_hierarchy!
                        ipu = self._source._keys_dict[pu]
                        self._source._embedding[iu,:] = self._source._embedding[ipu,:]

        n_to = len(self._target._keys)
        source_paths_s = np.array(source_paths_s, dtype='object') # avoid VisibleDeprecationWarning in shuffle
        self._rng.shuffle(source_paths_s)
        i = 0
        imax = steps * len(source_paths_s) - 1
        for step in range(steps):
            if debug_objective is not None:
                copy_long_key_embeddings()
                if step % int(debug_objective) == 0: # calculate objective every {debug_objective} step
                    loss, loss_neg, penalty_loss = self._calc_objective()
                    list_of_objectives.append(loss + negative * loss_neg + penalty * penalty_loss)
                    list_of_objectives_pos.append(loss)
                    list_of_objectives_neg.append(loss_neg)
                    list_of_objectives_penalty.append(penalty_loss)
            for u,iu in source_paths_s: # uniform sampling
                learning_rate = learning_rate_start + learning_rate_delta * (i/imax)
                v = self._gen.random_step(start=u, rng=self._rng)
                iv = self._target._keys_dict[v]
                self._update(iu, iv, 1, learning_rate) # 1-sigma(x) = sigma(-x)
                if self._neg_stationary: # stationary distribution
                    for v_neg in self._gen.random_step(start=(), size=negative, rng=self._rng):
                        iv_neg = self._target._keys_dict[v_neg]
                        self._update(iu, iv_neg, 0, learning_rate) # 0-sigma(x) = -sigma(x)
                else: # uniform distribution
                    for iv_neg in self._rng.integers(n_to, size=negative):
                        self._update(iu, iv_neg, 0, learning_rate) # 0-sigma(x) = -sigma(x)
                if penalty > 0:
                    self._update_hierarchy(u, iu, learning_rate, penalty, max_start_len)
                i+=1
        copy_long_key_embeddings()
        if debug_objective is not None:
            # finally, evaluate the objective once more.
            loss, loss_neg, penalty_loss = self._calc_objective()
            list_of_objectives.append(loss + negative * loss_neg + penalty * penalty_loss)
            list_of_objectives_pos.append(loss)
            list_of_objectives_neg.append(loss_neg)
            list_of_objectives_penalty.append(penalty_loss)
        self._total_steps += steps
        if debug_objective is not None:
            return dict(total_steps=self._total_steps, steps=steps, penalty=penalty, 
                negative=negative, neg_stationary=self._neg_stationary,
                learning_rate_start=learning_rate_start, learning_rate_end=learning_rate_start+learning_rate_delta,
                dimension=self.dimension, max_start_len=max_start_len, objectives=np.array(list_of_objectives),
                objectives_pos=np.array(list_of_objectives_pos), objectives_neg=np.array(list_of_objectives_neg),
                objectives_penalty=np.array(list_of_objectives_penalty))

    @property
    def config(self):
        "get configuration"
        cfg = dict(init_class=self.__class__.__name__, init_gen=self._gen._id, init_dimension=self._dimension, init_id=self._id, 
            init_seed=self._seed, init_neg_stationary=self._neg_stationary)
        # node_hierarchy is missing
        for i,th in enumerate(self._training_history):
            cfg.update({ f'train{i}_{k}':v for k,v in th.items() })
        return cfg