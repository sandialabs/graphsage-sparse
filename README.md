# GraphSAGE-Sparse
This is a refactor of the GraphSAGE algorithm (https://github.com/williamleif/GraphSAGE) that implements a sparse minibatch operation.  This is inspired by the sparse implementation of Spektral (https://github.com/danielegrattarola/spektral), except that only the subgraph associated with the minibatch is sent to the accelerator at each iteration.  This improves memory efficiency on very sparse graphs, and avoids the graph augmentation required by contemporary sampling approaches.

This code provides a TF 2.X implementation of the sparse algorithm (graphsage_sparse), as well as the original algorithm with neighbor sampling (graphsage_dense).  Both algorithms accept a sample_size parameter to control the maximum number of neighbors per node, but in the dense implementation this incurs a sample with replacement to ensure regularity of the input tensors, whereas the spare implementation does not have this constraint.  Support for the Open Graph Benchmark (https://ogb.stanford.edu/) and other datasets is provided for experimentation and comparisons.

## Implementation Variants
* graphage_dense
  * TF 2.X refactor that implements the batch sampling algorithm of https://arxiv.org/abs/1706.02216
  * As in the original implementation, both degree filtering and regularization are required
* graphsage_sparse
  * Sparse minibatch implementation - degree filtering is optional, and the graph is not regularized
  * Improved memory scaling, but requires longer initial training until data is cached 

## Datasets
* bter
  * Block Two-Level Erdos-Renyi graph (https://arxiv.org/abs/1112.3644)
* ogbn-arxiv
  * OGB Arxiv node prediction benchmark (https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)
  * Downloaded via Spektral on first use
* reddit
  * Preprocessed Reddit graph (http://snap.stanford.edu/graphsage/reddit.zip)
  * Downloaded via Spektral on first use

## Platforms
* gpu
  * NVIDIA Cuda
* gc
  * Graphcore IPU Pod (Poplar + TensorFlow), experimental

## Scripts
* python graphsage.py -n \<num\_devices\> -a \<algorithm\> -d \<dataset\> -p \<platform\>`
* algorithm:  one of `dense` or `sparse`
* dataset:  one of `bter`, `lreddit`, `nreddit`, or `arxiv`
* platform:  one of `gpu`, `gc`
* python graphsage.py --help to see more arguments

Examples of parameter sweep benchmarks are provided in \experiments, and notebooks to visualize results in \notebooks
