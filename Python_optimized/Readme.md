# Refactoring and Performance Optimization
As mentioned in appendix C of the thesis, the `EmbeddingView` adapts the different embeddings to a common interface. Since this interface proved useful, I decided to refactor the embeddings to provide two properties `source` and `target`, which essentially provide the same functionality.

However, for a HON Lattice 2D (§ 3.3), we need a `Lattice2D_EmbeddingView` instead of an `EmbeddingView`, as the former provides more functionality. Since the user no longer instantiates the `EmbeddingView` manually, we have to configure this somewhere - and the `HigherOrderPathGenerator` is a natural choice.
Unfortunately, this creates a cyclic dependency, because I use [typing hints](https://docs.python.org/3/library/typing.html) (without string forward references or `from __future__ import annotations`).
The solution was to create an `EmbeddingData`-class containing only the minimal functionality (without t-SNE visualization) and use dependency injection to configure which derived class to use.

The performance optimization consists of adding a new data structure to avoid dictionary lookups for random walk probabilities and using [Numba](https://numba.pydata.org/) (a JIT compiler for Python).

## How to use the code
The example from the main Readme needs small modifications:
```Python
from HigherOrderPathGenerator import HigherOrderPathGenerator
from Embedding import HON_DeepWalk_Embedding, HON_NetMF_Embedding
from Visualizations import create_EmbeddingData_Vis
gen = HigherOrderPathGenerator(..., create_EmbeddingData=create_EmbeddingData_Vis)
gen.load_BuildHON_rules(filename=..., freeze=True)

emb_D = HON_DeepWalk_Embedding(gen, dimension=128)
emb_D.train(num_walks=100, walk_length=80, window_size=10, num_iter=1)
print(emb_D.source.embedding)

emb_N = HON_NetMF_Embedding(gen, dimension=128, pairwise=True)
emb_N.train(window_size=5, negative=1)
print(emb_N.source.embedding) # or emb_N.target.embedding
```
Discarding the `EmbeddingView` is the most notable change
```Python
vis = emb_N.source.visualize_TSNE(random_state=random_state, title='TSNE, ' + emb_N._id)
vis.plot1() # hue='metadata'
```

## Changes
* After manipulating the rules of an `ABCHigherOrderPathGenerator`, call `freeze_rules()` to initialize source_paths, target_nodes, ...
  * Done in SyntheticNetworks.py and Datasets.py.
* Added class `EmbeddingData` to store the embedding vectors and related data - but nothing related to visualization.
  * `ABCHigherOrderPathGenerator` acts as a factory (using `create_source_embedding_data` and `create_target_embedding_data`)
  * Datasets.py and SyntheticNetworks.py configure this factory to use `EmbeddingData_Vis` and `EmbeddingData_Lattice2D`, respectively.
* Refactor Embeddings.py
  * `ABCSymmetricEmbedding`: replace `_embedding` by `_data._embedding`
  * `ABCAsymmetricEmbedding`: replace `_source_embedding` and `_target_embedding` by `_source._embedding` and `_target._embedding`, respectively
  * Tidy up the properties. (Commented code indicates, e.g., that the former `source_paths` is replaced by `source.keys`.)
* Removed `EmbeddingView` and `Lattice2D_EmbeddingView`
  * Functionality moved to `EmbeddingData_Vis` and `EmbeddingData_Lattice2D`, which inherit from `EmbeddingData`.
  * Replace `EmbeddingView(emb, use_source=True)` by `emb.source`; for use_source=False use `emb.target` instead.
  * `edge_distance` was set in the constructor of `Lattice2D_EmbeddingView` and is a property of `EmbeddingData_Lattice2D` now.
* Added OptimizedGenerator.py
  * Optimized with [Numba.jitclass](https://numba.pydata.org/numba-doc/dev/user/jitclass.html)
  * Used by `ABCHigherOrderPathGenerator` (`random_walks` [note the plural], `walk_probs`, and `walk_probs_df`)
  * Used by `HON_DeepWalk_Embedding`, `HON_Node2vec_Embedding`, `HON_NetMF_Embedding`, and `HON_GraRep_Embedding` and acivated by default (`use_numba=True`).