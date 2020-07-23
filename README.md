# Speeding up Word Mover's Distance and its variants via properties of distances between embeddings

This repository contains the source code used for all experiments.

Paper: https://arxiv.org/pdf/1912.00509.pdf


(under development) A [Python-wrapper package](https://github.com/matwerner/fast-wmd/tree/python-wrapper) of the main algorithms is also available.

# Environment tested

- Ubuntu 16.04 and 18.04
- Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz,with 8 GB of RAM

# Installing dependencies
```shell
sh install_dependencies.sh
```
It will install [Eigen3](http://eigen.tuxfamily.org/index.php) and [OR-Tools](https://developers.google.com/optimization/) dependencies.

# Building
```shell
sh build-project.sh
```
The application will be under build folder.

# Getting started

## Datasets

The datasets used durings this study can be obtained in 2 ways:

- Dropbox: https://www.dropbox.com/sh/9rnx8vwjvwjirsp/AADGwbT-aCqzZG7C_L1uNsY1a?dl=0 
    - Download each dataset and uncompress it inside the dataset folder.    
- Parse the datasets made available by Kusner in his [repository](https://github.com/mkusner/wmd)
    - Use the python scripts inside dataset folder
    - For AMAZON, BBCSPORT, CLASSIC, REUTERS and TWITTER use kusner_dataset_parser.py. E.g:    
    ```
    python3 kusner_dataset_parser.py amazon 0
    ```
    - For 20NEWS, OHSUMED and REUTERS use kusner_dataset_parser-2.py. E.g:    
    ```
    python3 kusner_dataset_parser.py 20news
    ```

## Application

Inside the build folder you will find fast-wmd executable.

### Kusner datasets

```shell
./fast-wmd kusner --tr <path-to-train-data> --te <path-to-test-data> --emb <path-to-embeddings> --k <k> --func <distance-function> --r <related-words> --verbose <true|false>
```
Options:

- **k**: number of neighbours from k-NN
- **func**: distance function to be used with k-NN. It can be:
    - **cosine**:      Cosine distance
    - **wmd**:         Word Mover's Distance (WMD)
    - **rwmd**:        Relaxed Word Mover's Distance (RWMD)
    - **lc-rwmd**:     Linear-Complexity Relaxed Word Mover's Distance (Following Atusu et al, 2017)
    - **wcd**:         Word Centroid Distance (WCD)
    - **rel-wmd**:     WMD + Edges reduction (Following Pele et al, 2009)
    - **rel-rwmd**:    RWMD + Edges reduction
    - **lc-rel-rwmd**: Linear-Complexity RWMD + Edges reduction
- **r**: number of related words per word. Used only by mf-wmd.
- **verbose**: dumps info about the distance computation between each document pair (Used for measuring distance function time)

### Triplets datasets

```shell
./fast-wmd triplets --triplets <path-to-test-set> --docs <path-to-test-data> --emb <path-to-embeddings> --num_clusters <number-of-clusters> --max_iter <number-of-iterations> --func <distance-function> --r <related-words> --verbose <true|false>
```

Options:
- **num_clusters**: number of clusters
- **max_iter**: Maximum number of iterations during clustering

After run it, it will dump in the console the configuration used, preprocessing and classification times and error rate.

## Scripts

As the number of experiments to be run is large, we implemented scripts for easily run most of them:

### For AMAZON, BBCSPORT, CLASSIC, RECIPE, TWITTER:
```shell
sh run-kusner-experiments.sh "<datasets>" "<partitions>" "<function>" "<r>" "<verbose>"
```

E.g:
```shell
sh run-kusner-experiments.sh "amazon bbcsport" "0 1 2 3 4" "rel-wmd" "1 2 4 8"
```
It will run the REL-WMD function with 1, 2, 4 and 8 related words for all partition of the AMAZON and BBCSPORTS datasets.

### For 20NEWS, OHSUMED, REUTERS:
```shell
sh run-kusner-experiments.sh "<datasets>" "<partitions>" "<function>" "<r>" "<verbose>"
```

E.g:
```shell
sh run-kusner-experiments.sh "20news" "wmd rwmd"
```
It will run the WMD and RWMD functions for the 20NEWS dataset.

### For ARXIV, WIKIPEDIA:
```shell
sh run-triplets-experiments.sh "<datasets>" "<function>" "<r>" "<num-clusters>" "<max-iterations>"
```

E.g:
```shell
sh run-triplets-experiments.sh "arxiv" "lc-rwmd lc-rel-rwmd" "16" "229" "5"
```

It will run the LC-RWMD and LC-REL-RWMD functions with r=16 for the ARXIV dataset, and applying a pre-processing step for clustering the embeddings with 229 clusters.
