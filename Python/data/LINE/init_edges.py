import os
import sys
sys.path.insert(0, r'..\..')
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from SyntheticNetworks import create_lattice_2nd_order_dynamic

latgen = create_lattice_2nd_order_dynamic(size=10, omega=0)
with open('edges.txt', 'w') as f:
    for path,prob in latgen.path_probs(start=(),num_steps=2):
        f.write(f'{path[0]} {path[1]} 1\n')