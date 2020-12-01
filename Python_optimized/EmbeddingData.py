from typing import Any, Callable, Dict, List, Iterator, Tuple, Optional
import numpy as np
import pandas as pd

class EmbeddingData(object):
    def __init__(self, emb, dimension: int, keys, keys_dict, use_source: bool, metadata = dict()):
        self._emb = emb
        self._dimension = dimension
        self._series = dict()
        self.use_source = use_source
        self._keys = keys
        self._keys_dict = keys_dict
        self.key2str = repr
        self._embedding = np.zeros(shape=(len(keys), dimension))
        for name, data in metadata.items():
            self.add_metadata(name, data)

    def add_metadata(self, name: str, data: Dict[Any,Any]):
        # data maps the keys to some grouping criterion
        self._series[name] = pd.Series({self.key2str(key): data[key] for key in self.keys})
        return self
    
    def __getitem__(self, series_name) -> pd.Series:
        "Get metadata by name as pandas Series"
        return self._series[series_name]

    def _append_metadata(self, data: pd.DataFrame, copy=False) -> pd.DataFrame:
        res = data.copy() if copy else data
        for name, s in self._series.items():
            res[name] = s
        return res

    def node2key(self, node:Any):
        "converts a node into a key"
        return (node,) if self.use_source else node
    
    @property
    def key2str(self):
        return self._key2str

    @key2str.setter
    def key2str(self, key2str):
        self._key2str = key2str
        self._keys_str = list(map(key2str, self._keys))

    @property
    def keys(self) -> List[Any]:
        "Keys used for the embedding"
        return self._keys
  
    @property
    def keys_str(self) -> List[str]:
        "String representation (i.e, self.key2str) of the keys. (Equals embedding.index)"
        return self._keys_str

    @property
    def embedding(self) -> pd.DataFrame:
        "Embedding, index corresponds to keys_str and columns are the dimensions o the embedding space."
        return pd.DataFrame(self._embedding, index=self._keys_str)

    def node_embedding_diff(self, node_pairs) -> np.ndarray:
        "For a given list of first oder edges (pairs of nodes), determine the difference in the embedding space between them"
        key_pairs = list((self.node2key(v), self.node2key(w)) for v,w in node_pairs)
        return self.key_embedding_diff(key_pairs)
    
    def key_embedding_diff(self, key_pairs) -> np.ndarray:
        "For a given list of pairs of keys, determine the difference in the embedding space between them"
        tmp = np.zeros(shape=(len(key_pairs), self._emb.dimension))
        i=0
        for start,end in key_pairs:
            e_start = self._embedding[self._keys_dict[start]] # = self.embedding.loc[key2str(start)]
            e_end = self._embedding[self._keys_dict[end]] # = self.embedding.loc[key2str(end)]
            tmp[i,:] = e_end - e_start
            i+=1
        return tmp

    @property
    def config(self):
        "get configuration"
        return dict(init_class=self.__class__.__name__, use_source=self.use_source)
    
    def write_config(self, file_object, comment: str = '', prefix: str = '', sep: str ='\t'):
        self._emb.write_config(file_object=file_object, comment=comment, prefix=prefix, sep=sep)
        file_object.write(prefix + '\n' + prefix + 'EmbeddingData:\n')
        for k,v in self.config.items():
            file_object.write(prefix + '%s%s%s\n' % (k,sep,v))