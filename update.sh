# This script builds a fresh distrobution of socialsim and installs it via pip.

python setup.py sdist
pip install dist/socialsim-0.1.0.tar.gz
