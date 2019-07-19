from setuptools import setup

package_structure = [
      'socialsim',
      'socialsim.measurements',
      'socialsim.measurements.model_parameters',
      'socialsim.visualizations'
]

package_requirements = [
      'pandas',
      'scipy>=1.2.1',
      'scikit-learn>=0.20.2',
      'fastdtw>=0.2.0',
      'pysal>=2.0.0',
      'tqdm>=4.31.1',
      'burst_detection>=0.1.0',
      'tsfresh>=0.11.2',
      'joblib>=0.13.2',
#      'networkx>=2.3',
#      'python-louvain>=0.13'
      'louvain>=0.6.1',
      'cairocffi>=1.0.2'
]

package_data = {
      'socialsim.measurements.model_parameters': ['best_model.pkl']
      }

setup(name='socialsim',
      version='0.2.1',
      packages=package_structure,
      package_data=package_data,
      license='',
      url='',
      long_description='None',
      maintainer='Zachary New',
      maintainer_email='zachary.new@pnnl.gov',
      install_requires=package_requirements
      )

