import numpy as np
from setuptools import setup, Extension, find_packages
from codecs import open
from os import path
import warnings


package_name = 'patternly'
example_dir = 'examples/'
example_data_dir = example_dir + 'example_data/'

version = {}
with open("version.py") as fp:
    exec(fp.read(), version)

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name=package_name,
    author='@zedlab_',
    author_email='ishanu@uchicago.edu',
    version = str(version['__version__']),
    packages=find_packages(),
    scripts=[],
    url='https://github.com/zeroknowledgediscovery/mantis',
    license='LICENSE',
    description='A tool for detecting anomalies in time series data',
    keywords=[
        'anomaly detection',
        'timeseries',
        'model-free',
        'adaptive',
    ],
    download_url='https://github.com/zeroknowledgediscovery/mantis/archive/'+str(version['__version__'])+'.tar.gz',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    install_requires=[
        "scikit-learn",
        "scipy",
        "numpy",
        "pandas",
        'zedsuite'
    ],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6"
    ],
    include_package_data=True,
)



