behavelet: a wavelet transform for mapping behavior
============
<p align="center">
<img src="https://github.com/jgraving/behavelet/blob/master/assets/wavelet.png" max-height:256px>
</p>

behavelet is a Python implementation of the normalized Morlet wavelet transform for behavioral mapping from [Berman et al. (2014)](https://doi.org/10.1098/rsif.2014.0672).

It runs on the CPU using numpy and multiprocessing or on the GPU using [CuPy](https://github.com/cupy/cupy).

This code was adapted to Python using [the original MotionMapper code](https://github.com/gordonberman/MotionMapper) from Gordon Berman et al.

Quick Start
------------
Here is an example of how to use behavelet on the CPU with a randomly generated dataset:
```python
from behavelet import wavelet_transform
import numpy as np

n_samples = 10000
n_features = 10
X = np.random.normal(size=(n_samples, n_features))

freqs, power, X_new = wavelet_transform(X, n_freqs=25, fsample=100., fmin=1., fmax=50.)
```
use the `n_jobs` argument to parallelize the computations across multiple threads:
```python
freqs, power, X_new = wavelet_transform(X, n_freqs=25, fsample=100., fmin=1., fmax=50., n_jobs=-1)
```
and use the `gpu` argument to run it on the GPU with [CuPy](https://github.com/cupy/cupy):
```python
freqs, power, X_new = wavelet_transform(X, n_freqs=25, fsample=100., fmin=1., fmax=50., gpu=True)
```
- `freqs` is a `(n_freqs,)` shaped array of the frequencies used for the wavelet transform
- `power` is a `(n_samples,)` shaped array with the total power for the wavelet coefficients in each sample
- `X_new` is a `(n_samples, n_freqs*n_features)` shaped array of the wavelet coefficients.

Citation
---------

If you use behavelet for your research, please cite our DOI:

[![DOI](https://zenodo.org/badge/204273245.svg)](https://zenodo.org/badge/latestdoi/204273245)

      @misc{graving2019behavelet,
            title={behavelet: a wavelet transform for mapping behavior},
            author={Graving, Jacob M},
            month={aug},
            year={2019},
            doi={10.5281/zenodo.3376742},
            url={https://doi.org/10.5281/zenodo.3376742}
            }

for the original description of the normalized Morlet wavelet transform see the paper from [Berman et al. (2014)](https://doi.org/10.1098/rsif.2014.0672):

    @article{berman2014mapping,
             title={Mapping the stereotyped behaviour of freely moving fruit flies},
             author={Berman, Gordon J and Choi, Daniel M and Bialek, William and Shaevitz, Joshua W},
             journal={Journal of The Royal Society Interface},
             volume={11},
             number={99},
             pages={20140672},
             year={2014},
             publisher={The Royal Society}
             }

Installation
------------
Install the latest stable version with pip:
```bash
pip install behavelet
```

Install the development version with pip:
```bash
pip install git+https://www.github.com/jgraving/behavelet.git
```

You can also install from within Python rather than using the command line, either from within Jupyter or another IDE, to ensure it is installed in the correct working environment:
```python
import sys
!{sys.executable} -m pip install git+https://www.github.com/jgraving/behavelet.git
```

If you wish to use the GPU version, you must [install CuPy manually](https://github.com/cupy/cupy#installation).

Development
-------------
Please submit bugs or feature requests to the [GitHub issue tracker](https://github.com/jgraving/behavelet/issues/new). Please limit reported issues to the behavelet codebase and provide as much detail as you can with a minimal working example if possible. 

If you experience problems with [CuPy](https://github.com/cupy/cupy), such as installing CUDA or other dependencies, then please direct issues to their development team.

Contributors
------------
behavelet was developed by [Jake Graving](https://github.com/jgraving), and is still being actively developed. Public contributions are welcome. If you wish to contribute, please [fork the repository](https://help.github.com/en/articles/fork-a-repo) to make your modifications and [submit a pull request](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).

License
------------
Released under a Apache 2.0 License. See [LICENSE](https://github.com/jgraving/behavelet/blob/master/LICENSE) for details.
