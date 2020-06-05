import os
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

try:
    __version__ = os.environ["GITHUB_REF"].split("/")[-1]
    print(f"Version: {__version__}")
except KeyError:
    from geneal.version import __version__

setup(
    name="geneal",
    version=__version__,
    description="Python Genetic Algorithms library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diogomatoschaves/geneal",
    author="Diogo Matos Chaves",
    author_email="di.matoschaves@gmail.com",
    packages=[*find_packages(), "geneal.utils"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
    install_requires=["numpy", "matplotlib", "scipy", "networkx", "pyturf", "pandas"],
    test_requires=["pytest", "pytest-cov", "pytest-mock"],
    keywords=["genetic algorithms", "ga", "optimization", "genetic programming"],
)
