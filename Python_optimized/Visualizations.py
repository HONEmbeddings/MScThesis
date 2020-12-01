from typing import Any, Callable, Dict, List, Iterator, Tuple, Optional
#from SyntheticNetworks import *
from EmbeddingData import EmbeddingData
from Embedding import ABCEmbedding, ABCSymmetricEmbedding
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
import numpy as np
import pandas as pd
import math
import os

class Visualization(object):
    def __init__(self, ev: "EmbeddingView", title, data, cols, config=dict(), edges=None):
        self._ev = ev
        self._title = title
        self._data = data
        self._cols = cols
        self.config = config
        self._edges = edges
        self._edge_args = dict(color='gray', linewidth=1, linestyle=':')
        self._figure = None
        self._figure_config = {}
    
    def __repr__(self):
        othercols = sorted(list(set(self._data.columns).difference(self._cols)))
        return f"""paths: {self.paths_id}
embedding: {self.emb_id}
dim: {self._data.shape}
embedding columns: {str(self._cols)}
other columns: {str(othercols)}"""
        
    @property
    def paths_id(self) -> str:
        "ID of the paths (transition probabilities)"
        return self._ev._emb._gen._id
    
    @property
    def emb_id(self) -> str:
        "ID of the embedding"
        return self._ev._emb._id + ('_source' if self._ev.use_source else '')
    
    @property
    def data(self) -> pd.DataFrame:
        "returns a DataFrame containing the (visualization of the) embedding togehter with explaining attributes."
        return self._data
    
    def _add_edges(self, ax, data, **kwargs):
        if self._edges is None:
            return
        data_xy = data[self._cols]
        lines = list([data_xy.loc[n1], data_xy.loc[n2]] for n1,n2 in self._edges)
        lc = LineCollection(lines, **kwargs)
        ax.add_collection(lc)

    def plot1(self, figsize: Tuple[float,float]=(6,4), dpi: int=200, figureargs=dict(), 
                 filter_col: Optional[str]=None, filter_values=set(), rotate:float=0,
                 return_figure=False, **kwargs):
        """
        Displays a visualization by a seaborn.scatterplot

        Parameters
        ----------
        figsize : (float,float)
            Width, height in inches.

        dpi : int, default=200
            Resolution of the figure.

        figureargs : dict
            These are passed to pyplot.figure

        filter_col : str, optional
            If specified, the data is filtered by this column, see filter_values.

        filter_values : set
            If filter_column is specified, consider only records where the column filter_col has values in filter_values.

        return_figure: bool, default=False
            If True, the figure is returned.

        kwargs :
            These are passed to seaborn.scatterplot (e.g. hue='col1', style='col2', alpha=0.5, palette='coolwarm').
        """
        data = self._data
        x_name, y_name = self._cols
        if rotate != 0: # rotate before edges
            data = data.copy()
            c,s = math.cos(rotate), math.sin(rotate)
            x = c*data[x_name] + s*data[y_name]
            y = -s*data[x_name] + c*data[y_name]
            data[x_name] = x
            data[y_name] = y
        fig = plt.figure(figsize=figsize, dpi=dpi, **figureargs)
        if self._title != '':
            fig.suptitle(self._title)
        self._add_edges(fig.gca(), data=data, **self._edge_args)
        palette = kwargs.pop('palette', 'coolwarm')
        if filter_col is not None: # apply filter after edges
            data = data[data[filter_col].isin(filter_values)]
        ax = sns.scatterplot(data=data, x=x_name, y=y_name, palette=palette, ax=fig.gca(), **kwargs)
        ax.set_aspect(1)
        # caution: by default there nothing to display in legend; should check for hue, style, ... in kwargs.
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        self._figure = fig
        self._figure_config = dict(figsize=figsize, dpi=dpi, figureargs=repr(figureargs), 
                filter_col=filter_col, filter_values=repr(filter_values), rotate=rotate, **kwargs)
        if return_figure:
            return fig
        
    def plot2(self, figsize: Tuple[float,float]=(15,10), dpi: int=200, figureargs=dict(), 
             filter_col: Optional[str]=None, filter_values=set(),
             return_figure=False, **kwargs):
        """
        Displays a visualization by two seaborn.scatterplots.
        The columns x_orig and y_orig added by Lattice2D_EmbeddingView._add_lattice_coord() are displayed as hue.

        Parameters
        ----------
        figsize : (float,float)
            Width, height in inches.

        dpi : int, default=200
            Resolution of the figure.

        figureargs : dict

        filter_col : str, optional
            If specified, the data is filtered by this column, see filter_values.

        filter_values : set
            If filter_column is specified, consider only records where the column filter_col has values in filter_values.

        return_figure: bool, default=False
            If True, the figure is returned.

        kwargs :
            These are passed to seaborn.scatterplot (e.g. style='key_len', alpha=0.5, palette='coolwarm').
        """
        data = self._data
        x_name, y_name = self._cols

        constrained_layout=figureargs.pop('constrained_layout',True)
        fig, axs = plt.subplots(1,2, figsize=figsize, dpi=dpi, **figureargs, constrained_layout=constrained_layout)
        if self._title != '':
            fig.suptitle(self._title)
        self._add_edges(axs[0], data, **self._edge_args)
        self._add_edges(axs[1], data, **self._edge_args)
           
        if filter_col is not None: # apply filter after edges
            data = data[data[filter_col].isin(filter_values)]

        palette = kwargs.pop('palette', 'coolwarm')
        scatterplot_args = dict(data=data, x=x_name, y=y_name, palette=palette, **kwargs)

        ax = sns.scatterplot(hue='x_orig', ax=axs[0], **scatterplot_args)
        ax.set_aspect(1)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

        ax = sns.scatterplot(hue='y_orig', ax=axs[1], **scatterplot_args)
        ax.set_aspect(1)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        self._figure = fig
        self._figure_config = dict(figsize=figsize, dpi=dpi, figureargs=repr(figureargs), 
                filter_col=filter_col, filter_values=repr(filter_values), **kwargs)
        if return_figure:
            return fig

    def annotate_node(self, node, text=None, ha='center', va='center', size=8):
        # ha in { 'left', 'center', 'right' }, va in { 'top', 'center', 'bottom' }
        ev = self._ev
        coord = self.data.loc[ev.key2str(ev.node2key(node)), self._cols]
        if text is None:
            text = str(node)
        for ax in self._figure.axes:
            ax.annotate(text, coord, horizontalalignment=ha, verticalalignment=va, size=size)

    def write_config(self, file_object, comment: str = '', prefix: str = '', sep: str ='\t', write_figure_config: bool = True):
        self._ev.write_config(file_object=file_object, comment=comment, prefix=prefix, sep=sep)
        file_object.write(prefix + '\n' + prefix + 'Visualization:\n')
        for k,v in self.config.items():
            file_object.write(prefix + '%s%s%s\n' % (k,sep,v))
        if write_figure_config:
            file_object.write(prefix + '\n' + prefix + 'Plot:\n')
            for k,v in self._figure_config.items():
                file_object.write(prefix + '%s%s%s\n' % (k,sep,v))

    def save_describe(self, filename, comment : str = '', bbox_inches='tight', **kwargs):
        "saves the most recent figure created by this instance and exports the configs to a text file."
        self._figure.savefig(filename, bbox_inches=bbox_inches, **kwargs)
        with open(filename + '.txt','w') as f:
            self.write_config(f, comment=comment, prefix='', sep='\t', write_figure_config=True)

def create_EmbeddingData_Vis(*args, **kwargs):
    return EmbeddingData_Vis(*args, **kwargs)

class EmbeddingData_Vis(EmbeddingData):
    def __init__(self, emb, dimension: int, keys, keys_dict, use_source: bool, metadata = dict()):
        super().__init__(emb, dimension, keys, keys_dict, use_source, metadata)

    def visualize_TSNE(self, random_state=1, n_iter=1000, title: Optional[str]=None, **kwargs) -> Visualization:
        "Returns TSNE visualization"
        config = dict(creator='visualize_TSNE', random_state=random_state, n_iter=n_iter, title=title, **kwargs)
        embedding = self.embedding
        cols = ['x','y']
        data = TSNE(n_components=2, random_state=random_state, n_iter=n_iter, **kwargs).fit_transform(embedding)
        data = pd.DataFrame(data, index=embedding.index, columns=cols)
        if title is None:
            title = 'TSNE(random_state=%d)' % random_state
        return Visualization(self, title, self._append_metadata(data), cols, config)

def create_EmbeddingData_Lattice2D(*args, **kwargs):
    return EmbeddingData_Lattice2D(*args, **kwargs)

class EmbeddingData_Lattice2D(EmbeddingData):
    def __init__(self, emb, dimension: int, keys, keys_dict, use_source: bool, metadata = dict()):
        super().__init__(emb, dimension, keys, keys_dict, use_source, metadata)
        self.creator = emb._gen.creator # todo: check has_attr and type
        self.edge_distance = 1

    @property
    def edge_distance(self):
        return self._edge_distance

    @edge_distance.setter
    def edge_distance(self, value):
        self._edge_distance = value
        if value is None:
            self._hor_edge_pairs = []
            self._ver_edge_pairs = []
        elif value == 1:
            self._hor_edge_pairs = self.creator.horizontal_edges1
            self._ver_edge_pairs = self.creator.vertical_edges1
        else:
            self._hor_edge_pairs = list((k1[0],k2[1]) for k1,k2 in self.creator.horizontal_edges2)
            self._ver_edge_pairs = list((k1[0],k2[1]) for k1,k2 in self.creator.vertical_edges2)
    
    def visualize_proj(self, disp_lengths: bool=True, title: Optional[str]=None) -> Visualization:
        """
        Returns a projection visualization.
        The projection is determined utilizing knowledge about lattice structure (i.e. 'right' and 'down' neighbors).
        """
        config = dict(creator='visualize_proj', disp_lengths=disp_lengths, title=title)
        # Since TSNE may distort the data, find average directions of horizontal and vertical edges
        max_key_len = max(len(key) for key in self._keys) if self.use_source else 1
        if max_key_len == 1:
            hor_edge_diffs = self.node_embedding_diff(self.creator.horizontal_edges1)
            ver_edge_diffs = self.node_embedding_diff(self.creator.vertical_edges1)
        else:
            hor_edge_diffs = self.key_embedding_diff(self.creator.horizontal_edges2)
            ver_edge_diffs = self.key_embedding_diff(self.creator.vertical_edges2)
        if disp_lengths:
            hor_edge_lengths = np.linalg.norm(hor_edge_diffs, axis=1)
            ver_edge_lengths = np.linalg.norm(ver_edge_diffs, axis=1)
            print('max_key_len', max_key_len)
            print('average horizontal edge length', hor_edge_lengths.mean())
            print('average vertical edge length', ver_edge_lengths.mean())
            print('ratio', hor_edge_lengths.mean()/ver_edge_lengths.mean())
        mean_hor_edge = hor_edge_diffs.mean(axis=0)
        mean_ver_edge = ver_edge_diffs.mean(axis=0)
        embedding = self.embedding
        cols = ['proj_hor','proj_ver']
        emb_proj_hor = embedding @ (mean_hor_edge / np.linalg.norm(mean_hor_edge))
        emb_proj_ver = embedding @ (mean_ver_edge / np.linalg.norm(mean_ver_edge))
        data = pd.DataFrame(np.array([emb_proj_hor, emb_proj_ver]).T, index=embedding.index, columns=cols)
        if title is None:
            title = 'Projection along average horizontal & vertical edge'
        edges = self._get_edges()
        return Visualization(self, title, self._append_metadata(data), cols, config, edges)

    def visualize_lattice(self, title: Optional[str]=None) -> Visualization:
        "Displays the coordinates of the embeddings"
        config = dict(creator='visualize_lattice', title=title)
        data = pd.DataFrame({'x': self._series['x_orig'], 'y': self._series['y_orig']})
        cols = data.columns.tolist()
        edges = self._get_edges()
        return Visualization(self, title, self._append_metadata(data), cols, config, edges)

    def _auto_transform_orthogonal(self, vis_data, verbose=False) -> pd.DataFrame:
        """
        Visualizations are often defined in terms of relative distances, 
        which are not affected by orthogonal transformations.
        Givent the ground truth is known, let us try to align the visualization.
        """
        # We would like to use horizontal/vertical edges to determione the direction of the x and y axis.
        # The edges should be oriented from left to right and from down to up.
        # However, some embeddings separate the nodes into two clusters (even and odd).
        # Therefore, take horizontal and vertical edges of distance 2 unless specified otherwise.
        def vis_diff(vis_data, node_pairs):
            tmp = np.zeros(shape=(len(node_pairs), vis_data.shape[1]))
            i=0
            for start,end in node_pairs:
                v_start = vis_data.loc[self.key2str(self.node2key(start))]
                v_end = vis_data.loc[self.key2str(self.node2key(end))]
                tmp[i,:] = v_end - v_start
                i += 1
            return tmp
        hor_edge_diffs = vis_diff(vis_data, self._hor_edge_pairs)
        ver_edge_diffs = vis_diff(vis_data, self._ver_edge_pairs)
        hor_edge_mean = hor_edge_diffs.mean(axis=0)
        ver_edge_mean = ver_edge_diffs.mean(axis=0)
        det = np.linalg.det(np.matrix([hor_edge_mean,ver_edge_mean]))
        if verbose:
            print(hor_edge_mean, ver_edge_mean, det)
        if det < 0:
            y_col = vis_data.columns[1]
            vis_data = vis_data.copy()
            vis_data[y_col] = -vis_data[y_col]
            hor_edge_mean[1] *= -1
            ver_edge_mean[1] *= -1
        # determine the angle by which hor_edge_mean should be rotated to point towards [1,0].
        angle1 = -math.atan2(hor_edge_mean[1], hor_edge_mean[0])
        c, s = math.cos(angle1), math.sin(angle1)
        rot = np.matrix([[c,-s],[s,c]])
        if verbose:
            print('1st rotation by', angle1, 'results in', (rot @ hor_edge_mean).A1, (rot @ ver_edge_mean).A1)

        # Determine the angle by which ver_edge_mean should be rotated to point towards [0,1].
        # To avoid calculations modulo 2*pi estimate the angle2 after applying the rotation by angle1.
        # Checking the determinant already ensures that angle2 is in [-pi/2, pi/2], and I expect angle2 to be small.
        rot_ver_edge_mean = (rot @ ver_edge_mean).A1
        angle2 = math.atan2(rot_ver_edge_mean[0], rot_ver_edge_mean[1])
        if verbose:
            print('angle2', angle2)
        angle = angle1 + angle2/2 # take average
        c,s = math.cos(angle),math.sin(angle)
        rot = np.matrix([[c,-s],[s,c]])
        if verbose:
            print('2nd rotation by', angle, 'results in', (rot @ hor_edge_mean).A1, (rot @ ver_edge_mean).A1)
        return pd.DataFrame(vis_data.values @ rot.T, index=vis_data.index, columns=vis_data.columns)

    def _get_edges(self):
        if self._edge_distance is None:
            return None
        edges = self._hor_edge_pairs + self._ver_edge_pairs
        # convert pairs of nodes into pairs of string representations of keys
        return list((self.key2str(self.node2key(n1)),self.key2str(self.node2key(n2))) for n1,n2 in edges)

    def visualize_TSNE(self, random_state=1, n_iter=1000, title: Optional[str]=None, 
            autotransform: bool = True, autotransform_verbose:bool = False, **kwargs) -> Visualization:
        "Returns TSNE visualization"
        config = dict(creator='visualize_TSNE', random_state=random_state, n_iter=n_iter, title=title, 
                autotransform=autotransform, **kwargs)
        embedding = self.embedding
        cols = ['x','y']
        data = TSNE(n_components=2, random_state=random_state, n_iter=n_iter, **kwargs).fit_transform(embedding)
        data = pd.DataFrame(data, index=embedding.index, columns=cols)
        if title is None:
            title = 'TSNE(random_state=%d)' % random_state
        if autotransform:
            data = self._auto_transform_orthogonal(data, verbose=autotransform_verbose)
        edges = self._get_edges()
        return Visualization(self, title, self._append_metadata(data), cols, config, edges)


    def _calc_embedding_edges_len_angle(self, horizontal: bool):
        "Calculates lengths and angles (to their average) of embedded edges"
        edges = self._hor_edge_pairs if horizontal else self._ver_edge_pairs
        keys = list(f'{n1} -> {n2}' for  n1,n2 in edges)
        emb_edges = self.node_embedding_diff(edges)
        mean_edge = emb_edges.mean(axis=0)
        mean_edge1 = mean_edge / np.linalg.norm(mean_edge)
        edges_len = np.linalg.norm(emb_edges,axis=1)
        edges_spr = (emb_edges @ mean_edge1) / edges_len
        edges_angle = np.arccos( edges_spr.clip(-1,1) )
        return pd.DataFrame({'len': edges_len, 'angle': edges_angle, 'angle360': edges_angle *180/math.pi, 
                'direction': ('horizontal' if horizontal else 'vertical')}, index=keys)

    def visualize_edges_len_angle(self, title: Optional[str]=None,
            figsize: Tuple[float,float]=(6,4), dpi: int=200, figureargs=dict(), **kwargs):
        config=dict(creator='visualize_edges_len_angle')
        hor_stats = self._calc_embedding_edges_len_angle(True)
        ver_stats = self._calc_embedding_edges_len_angle(False)
        stats = hor_stats.append(ver_stats)
        stats.sort_values('len', inplace=True) # in case of overplotting, treat horizontal and vertical edges equally
        cols = ['len','angle']
        if title is None:
            title = 'Edge lengths and angles'
        # This plot serves only to illustrate conceptual limitations of visualize_proj(),
        # which does not justify changes in class Visualization.
        # As a workaround, return an instance of Visualization, which has already created a figure.
        vis = Visualization(self, title, stats, cols, config, edges=None)
        vis.plot1(figsize=figsize, dpi=dpi, hue='direction', figureargs=figureargs, **kwargs) 
        ax = vis._figure.gca()
        ax.set_aspect('auto')
        ax.set_xlim(0, None)
        ax.set_ylim(0, math.pi)
        ax.tick_params(labelright=True, right=True)
        ax.set_yticks(math.pi * np.linspace(0,1,num=5))
        ax.set_yticklabels(['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$'])
        ax.hlines(math.pi/2, *ax.get_xlim(),linewidths=0.5)
        ax.legend(loc='upper left')
        vis._figure_config['workaround'] = 'see visualize_edges_len_angle' # let save_describe() know about this workaround
        return vis

    @property
    def config(self):
        "get configuration"
        cfg = super().config
        cfg['edge_distance'] = self._edge_distance
        if self._edge_distance is not None:
            cfg['mean horizontal edge length'] = self._calc_embedding_edges_len_angle(True)['len'].mean()
            cfg['mean vertical edge length'] = self._calc_embedding_edges_len_angle(False)['len'].mean()
            cfg['stretch'] = cfg['mean vertical edge length'] / cfg['mean horizontal edge length']
        return cfg


## ---- compatibility ----
# class EmbeddingView(object):
#     "Helper class for visualizing embeddings"
#     def __init__(self, emb : ABCEmbedding, use_source: bool = True):
#         self._emb = emb
#         if isinstance(emb, ABCSymmetricEmbedding):
#             self._data = emb._data
#         elif use_source:
#             self._data = emb.source
#         else:
#             self._data = emb.target
    
#     def add_metadata(self, name, data):
#         self._data.add_metadata(name, data)
#         return self

#     def __getitem__(self, series_name) -> pd.Series:
#         return self._data[series_name]

#     def _append_metadata(self, data: pd.DataFrame, copy=False) -> pd.DataFrame:
#         return self._data._append_metadata(data, copy)
    
#     def node2key(self, node:Any):
#         "converts a node into a key"
#         return self._data.node2key(node)

#     @property
#     def keys(self): # -> List[Any]:
#         "Keys used for the embedding"
#         return self._data.keys
  
#     @property
#     def keys_str(self): # -> List[str]:
#         "String representation (i.e self.key2str) of the keys. (Equals embedding.index)"
#         return self._data.keys_str
    
#     @property
#     def embedding(self) -> pd.DataFrame:
#         "Embedding, index corresponds to keys_str and columns are the dimensions o the embedding space."
#         return self._data.embedding

#     @property
#     def use_source(self):
#         return self._data.use_source

#     # By default, self._data does not implement visualize_TSNE.
#     def visualize_TSNE(self, random_state=1, n_iter=1000, title: Optional[str]=None, **kwargs) -> Visualization:
#         "Returns TSNE visualization"
#         config = dict(creator='visualize_TSNE', random_state=random_state, n_iter=n_iter, title=title, **kwargs)
#         embedding = self.embedding
#         cols = ['x','y']
#         data = TSNE(n_components=2, random_state=random_state, n_iter=n_iter, **kwargs).fit_transform(embedding)
#         data = pd.DataFrame(data, index=embedding.index, columns=cols)
#         if title is None:
#             title = 'TSNE(random_state=%d)' % random_state
#         return Visualization(self, title, self._append_metadata(data), cols, config)

#     @property
#     def config(self):
#         "get configuration"
#         return self._data.config
    
#     def write_config(self, file_object, comment: str = '', prefix: str = '', sep: str ='\t'):
#         self._emb.write_config(file_object=file_object, comment=comment, prefix=prefix, sep=sep)
#         file_object.write(prefix + '\n' + prefix + 'EmbeddingView:\n')
#         for k,v in self.config.items():
#             file_object.write(prefix + '%s%s%s\n' % (k,sep,v))

# class Lattice2D_EmbeddingView(EmbeddingView):
#     "EmbeddingView for the synthetic network 'Lattice2D_2nd_order_dynamic'"
#     def __init__(self, emb : ABCEmbedding, use_source: bool = True, edge_distance: Optional[int] = 1):
#         super().__init__(emb, use_source)
#         self.creator = self._data.creator # todo: check has_attr and type
#         self._data.edge_distance = edge_distance

#     def visualize_proj(self, disp_lengths: bool=True, title: Optional[str]=None) -> Visualization:
#         """
#         Returns a projection visualization.
#         The projection is determined utilizing knowledge about lattice structure (i.e. 'right' and 'down' neighbors).
#         """
#         return self._data.visualize_proj(disp_lengths=disp_lengths, title=title)

#     def visualize_lattice(self, title: Optional[str]=None) ->  Visualization:
#         "Displays the coordinates of the embeddings"
#         return self._data.visualize_lattice(title=title)

#     def visualize_TSNE(self, random_state=1, n_iter=1000, title: Optional[str]=None, 
#             autotransform: bool = True, autotransform_verbose:bool = False, **kwargs) -> Visualization:
#         return self._data.visualize_TSNE(random_state=random_state, n_iter=n_iter, title=title,
#             autotransform=autotransform, autotransform_verbose=autotransform_verbose, **kwargs)

#     def visualize_edges_len_angle(self, title: Optional[str]=None,
#             figsize: Tuple[float,float]=(6,4), dpi: int=200, figureargs=dict(), **kwargs) ->  Visualization:
#         return self._data.visualize_edges_len_angle(title=title, figsize=figsize, dpi=dpi, figureargs=figureargs, **kwargs)

#     def node_embedding_diff(self, node_pairs):
#         return self._data.node_embedding_diff(node_pairs)

## Usage:
# from SyntheticNetworks import create_lattice_2nd_order_dynamic
# from Embedding import HON_DeepWalk_Embedding
# latgen = create_lattice_2nd_order_dynamic(size=10, omega=0.5, lattice_sep='-')
# emb = HON_DeepWalk_Embedding(latgen, 128)
# %time emb.train(random_seed=1)
# vis = emb.source.visualize_TSNE(random_state=1, n_iter=1000)
# vis.plot1(hue='even_odd', style='direction', rotate=2.4)
# vis.plot2(style='direction', alpha=0.5)
#
# only_len_1 = dict(filter_col='key_len', filter_values={1})
# only_len_2 = dict(filter_col='key_len', filter_values={2})
# vis.plot2(style='direction', alpha=0.5, **only_len_2)