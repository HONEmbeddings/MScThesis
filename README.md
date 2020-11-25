# Application of Higher-Order Network Models to Representation Learning in Sequential Data
This repository provides the Python implementation for my master thesis.

## Setting up the Python environment
The software has been developped with [Anaconda](https://www.anaconda.com/). Just install the default packages plus `gensim` (which requires `smart_open`):
```
conda create --name thesis python=3.7 anaconda gensim smart_open
```

A minimal environment is created with:
```
conda create --name thesis python=3.7 pandas numpy scipy scikit-learn matplotlib seaborn tqdm notebook ipywidgets gensim smart_open 
```

The notebooks in `Python/data/models` additionally require [pathpy2](https://pypi.org/project/pathpy2/)
```
pip install pathpy2
```

## How to use the code
The code is split between Python files and Jupyter notebooks, where the former contain shared code used by the latter.

First, we need to initialize a `HigherOrderPathGenerator` with the probabilities.
The transition probabilities associated with an empty tuple of nodes should correspond to the first-order stationary distribution. In `Python/data/models/SocioPatterns.ipynb`, such probabilities are extracted from time-stamped data and exported to text files.
Besides the probabilities, we may also define attributes for the individual nodes (i.e., metadata) and the dataset (i.e., config); see `Datasets.py` for an example.

After initializing the `HigherOrderPathGenerator`, proceed as in `Classification_sim.ipynb`.

To obtain the embeddings for HON DeepWalk or HON NetMF as Pandas DataFrames, we use:
```Python
gen = HigherOrderPathGenerator()
gen.load_BuildHON_rules(filename=...)

emb_D = HON_DeepWalk_Embedding(gen, dimension=128)
emb_D.train(num_walks=100, walk_length=80, window_size=10, num_iter=1)
print(emb_D.embedding)

emb_N = HON_NetMF_Embedding(gen, dimension=128, pairwise=True)
emb_N.train(window_size=5, negative=1)
print(emb_N.source_embedding) # or emb_N.target_embedding
```

Note the different properties for the embedding data (embedding, source_embedding, and target_embedding).
Hence, we wrap the Embedding instance into an `EmbeddingView` to select and visualize the desired embedding.
```Python
#ev = EmbeddingView(emb_D)
ev = EmbeddingView(emb_N, use_source=True)
vis = ev.visualize_TSNE(random_state=random_state, title='TSNE, ' + ev._emb._id)
vis.plot1() # hue='metadata'
```

## Citation

```
@MastersThesis{Studer2020,
  author = {Michael Markus Studer},
  title = {{Application of Higher-Order Network Models to Representation Learning in Sequential Data}},
  school = {University of Zurich},
  address = {Switzerland},
  year = {2020},
}
```