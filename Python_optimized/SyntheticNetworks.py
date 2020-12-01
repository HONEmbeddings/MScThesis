from HigherOrderPathGenerator import HigherOrderPathGenerator
from Visualizations import create_EmbeddingData_Lattice2D

from collections import Counter
import numpy as np
import pandas as pd
import math

def create_lattice_2nd_order_dynamic(size: int=10, omega: float=0, lattice_sep: str='-', check: bool=False):
    creator = Lattice2D_2nd_order_dynamic(size, lattice_sep)
    return creator.create_generator(omega, check)

class Lattice2D_2nd_order_dynamic(object):
    def __init__(self, size: int=10, lattice_sep: str='-'):
        self.size=size
        self.lattice_sep = lattice_sep
        # '0-0' corresponds to the lower-left corner of the lattice. Format is (x,y).
        self.neighbor_funcs = { (-1,0): self.left, (1,0): self.right, (0,1): self.up, (0,-1): self.down }
        self.coord2nodes = { (x,y): '%i%s%i' % (x, lattice_sep, y) for x in range(size) for y in range(size) }
        self.node2coords = { n:c for c,n in self.coord2nodes.items() }

    def left(self, node:str)->str:
        x,y = node.split(self.lattice_sep)
        x = str(max(0, int(x)-1))
        return x + self.lattice_sep + y
    def right(self, node:str)->str:
        x,y = node.split(self.lattice_sep)
        x = str(min(self.size-1, int(x)+1))
        return x + self.lattice_sep + y
    def up(self, node:str)->str:
        x,y = node.split(self.lattice_sep)
        y = str(min(self.size-1, int(y)+1))
        return x + self.lattice_sep + y
    def down(self, node:str)->str:
        x,y = node.split(self.lattice_sep)
        y = str(max(0, int(y)-1))
        return x + self.lattice_sep + y

    @property
    def horizontal_edges1(self):
        return list( (u,v) for (u,v) in [(u,self.right(u)) for u in self.coord2nodes.values()] if u!=v )

    @property
    def vertical_edges1(self):
        return list( (u,v) for (u,v) in [(u,self.up(u)) for u in self.coord2nodes.values()] if u!=v )

    @property
    def horizontal_edges2(self):
        return list( ((u,v),(v,w)) for (u,v,w) in [(self.left(u),u,self.right(u)) for u in self.coord2nodes.values()] if u!=v and v!=w)

    @property
    def vertical_edges2(self):
        return list( ((u,v),(v,w)) for (u,v,w) in [(self.down(u),u,self.up(u)) for u in self.coord2nodes.values()] if u!=v and v!=w)

    def create_generator(self, omega: float=0, check: bool=False):
        assert abs(omega)<=1
        config = dict(code=__file__, init=self.__class__.__name__ + '.create_generator',
                      size=self.size, lattice_sep=repr(self.lattice_sep), omega=omega)
        # create list of nodes and (undirected) edges
        # see also pathpy.generators.lattice_network(start=0, stop=size, dims=2)
        nodes = list(self.coord2nodes.values())
        undirected_edges = self.horizontal_edges1 + self.vertical_edges1
        edges = undirected_edges + [(v,u) for u,v in undirected_edges]
        degrees = Counter(u for u,v in edges)
        sum_degrees = sum(degrees.values())

        #node_sort_key = lambda x: tuple(map(int,x.split('-')))
        node_sort_key = self.node2coords.__getitem__
        latgen = HigherOrderPathGenerator(node_sort_key=node_sort_key, id='Lattice2D(%d, omega=%f)' % (self.size, omega), 
            config=config, create_EmbeddingData=create_EmbeddingData_Lattice2D)
        latgen.creator = self
        latgen.load_edge_list(edges, directed=True)
        # add 2nd order rules for (u,v) -> w
        for u in nodes:
            latgen.add_rule((), u, degrees[u]/sum_degrees) # prob of stationary distribution
            for (d1,f1) in self.neighbor_funcs.items():
                v = f1(u)
                if u == v: # cannot move / no self-loops
                    continue
                for (d2,f2) in self.neighbor_funcs.items():
                    w = f2(v)
                    if v == w: # cannot move / no self-loops
                        continue
                    if v == f1(v): # if we cannot move twice in direction f1 from u, do not change dynamic
                        d = 0.0
                    else:
                        d = d1[0]*d2[0] - d1[1]*d2[1] # horizontal speed-up and vertival slow-down
                    latgen.add_rule((u, v), w, (1 + d*omega)/degrees[v])
        latgen.freeze_rules()

        latgen.add_source_path_metadata('key_len', { n: len(n) for n in latgen.source_paths }, check=False)
        latgen.add_metadata('x_orig', { n: c[0] for c,n in self.coord2nodes.items() }, use_last=True, check=False)
        latgen.add_metadata('y_orig', { n: c[1] for c,n in self.coord2nodes.items() }, use_last=True, check=False)
        latgen.add_metadata('parity', { n: 'even' if sum(c)%2==0 else 'odd' for c,n in self.coord2nodes.items() }, use_last=True, check=False)
        direction = dict()
        for key in latgen.source_paths:
            if len(key) == 1:
                direction[key]='none'
            else:
                c0 = self.node2coords[key[0]]
                c1 = self.node2coords[key[-1]]
                delta = (c1[0]-c0[0], c1[1]-c0[1])
                direction[key]=self.neighbor_funcs[delta].__name__
        latgen.add_source_path_metadata('direction', direction, check=False)
        if check:
            # check whether probabilities sum up to one
            print(list(latgen.check_transition_probs(tol=1e-14)))
            # verify that the 2nd order rules did not modify the stationary distribution
            print(list(latgen.verify_stationarity(order=2, tol=1e-17)))
        return latgen
