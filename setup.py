from setuptools import setup, find_packages
from os import path
import numpy as np

this_directory = path.abspath(path.dirname(__file__))

version = {}
with open("mantis/_version.py") as fp:
    line = fp.read()
    #print(line)
    ls = line.split('.')
    last = str(int(ls[-1].split('"')[0]) + 1) + '"'
    ls[-1] = last
    #print('.'.join(ls))
    ls[-1] = last
    exec(line, version)

# with open("mantis/_version.py", "w") as fp:
# 	fp.write('.'.join(ls))

with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mantis',
    version=version['__version__'],
    packages=find_packages(),
    keywords=['timeseries'],
    install_requires=['pandas', 'numpy', 'scikit-learn', 'zedsuite'],
    # metadata for PyPI upload
    url='https://github.com/zeroknowledgediscovery/mantis',
    description=("A package for time series anomaly detection"),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
)


