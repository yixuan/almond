from setuptools import setup, find_packages

setup(
    name="almond",
    version="0.1.0",
    packages=find_packages(),
    author="Yixuan Qiu",
    author_email="yixuan.qiu@cos.name",
    description="ALMOND: Adaptive latent modeling and optimization via neural networks and Langevin diffusion",
    install_requires=[
        # "mxnet", -> let the user install mxnet manually, as there are so many versions
        "numpy"
    ]
)
