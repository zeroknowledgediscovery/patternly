from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
package_name = 'mantis'

version = {}
with open('version.py') as fp:
    exec(fp.read(), version)

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=package_name,
    author='zed.uchicago.edu',
    author_email='ishanu@uchicago.edu',
    version = str(version['__version__']),
    packages=find_packages(),
    scripts=[],
    url='https://github.com/zeroknowledgediscovery/mantis',
    license='LICENSE',
    description='A package for time series anomaly detection',
    keywords=[
        'time series',
        'anomaly detection',
        'machine learning',
    ],
    download_url='https://github.com/zeroknowledgediscovery/mantis/archive/'+str(version['__version__'])+'.tar.gz',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    install_requires=[
        'numpy',
        'pandas',
        'sklearn',
        'zedsuite',
    ],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'],
    include_package_data=True,
)
