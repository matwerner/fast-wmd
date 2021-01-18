# Speeding up Word Mover's Distance and its variants via properties of distances between embeddings

This repository is a python-wrapper of the main algorithms implemented during this work.

Algorithms implemented:

- WMD, RWMD (standard)
- Rel-WMD, Rel-RWMD (standard)

TODO:
- Linear versions of RWMD and Rel-RWMD;
- Code documentation (DONE);
- Package documentation;
- Error handling;
- Examples using jupyter notebook
- Publish package in PyPi;
- ...

Nevertheless, an example is available in the jupyter notebook folder.

# Installing dependencies
```shell
sh install_dependencies.sh
```
It will install [Eigen3](http://eigen.tuxfamily.org/index.php) and [OR-Tools](https://developers.google.com/optimization/) dependencies.

# Building
```shell
cd wrapper
python3 setup.py build_ext --inplace
```
