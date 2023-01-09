from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as rdm:
    long_description = rdm.read()

with open("src/whitespace_correction/version.py", "r", encoding="utf8") as vf:
    version = vf.readlines()[-1].strip().split()[-1].strip("\"'")

setup(
    name="whitespace_correction",
    version=version,
    description="Correct missing or spurious whitespaces in text.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://whitespace-correction.cs.uni-freiburg.de",
    download_url="https://pypi.org/project/whitespace-correction",
    project_urls={
        "Website": "https://whitespace-correction.cs.uni-freiburg.de",
        "Github": "https://github.com/ad-freiburg/whitespace-correction",
    },
    author="Sebastian Walter",
    author_email="swalter@cs.uni-freiburg.de",
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={
        "console_scripts": ["wsc=whitespace_correction.api.cli:main"],
    },
    install_requires=[
        "torch>=1.8.0",
        "einops>=0.3.0",
        "numpy>=1.19.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.49.0",
        "requests>=2.27.0",
        "flask>=2.0.0",
    ],
    extras_require={
        "train": [
            "tensorboard>=2.8.0",
        ],
    }
)
