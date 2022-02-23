from distutils.core import setup

setup(
    name="trt",
    version="0.1.0",
    description="Tokenization repair using Transformers",
    author="Sebastian Walter",
    author_email="swalter@tf.uni-freiburg.de",
    packages=[
        "trt"
    ],
    install_requires=[
        "torch>=1.8.0",
        "einops>=0.3.0",
        "numpy>=1.19.0",
        "tokenizers>=0.10.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.49.0"
    ],
    extras_require={
        "train": [
            "lmdb>=1.1.0",
            "msgpack>=1.0.0",
            "catalyst>=20.12",
            "tensorboard>=2.8.0",
            "gputil>=1.4.0",
        ]
    }
)
