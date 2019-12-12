from setuptools import setup

import json 

package_structure = [
    'socialsim_scoring'
    ]

requirements = [
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn'
    ]

with open('socialsim_scoring/_version.json') as f:
    config = json.load(f)
    __version__ = config["version"]

setup(
    name='socialsim_scoring', 
    version=__version__, 
    packages=package_structure, 
    license='NA',
    url='NA',
    long_description='NA',
    maintainer='Zachary New',
    maintainer_email='zachary.new@pnnl.gov',
    install_requires=requirements
    )
