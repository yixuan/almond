## ALMOND <img src="https://statr.me/images/sticker-almond.png" alt="almond" height="150px" align="right" />

This repository stores the code files for the article
"[ALMOND: Adaptive Latent Modeling and Optimization via Neural Networks and Langevin Diffusion](https://doi.org/10.1080/01621459.2019.1691563)".

### Installation

ALMOND depends on the [MXNet](https://mxnet.incubator.apache.org/) deep learning framework.
First install a proper version of MXNet following the
[official documentation](https://mxnet.incubator.apache.org/get_started), for example:

```bash
pip install mxnet --user
```

And then enter the `package` directory and run

```bash
cd package
python3 setup install --user
```

### Running Examples

The `experiments` directory contains the code files for all numerical experiments in the
article. Simply run the Python/R files in order in each subdirectory. `results` contains
the generated plots used by the article.

### License

The `almond` Python package and the experiment code files are under the MIT License.
