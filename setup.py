from distutils.core import setup

package_structure = ['socialsim',
                     'socialsim.measurements',
                     'socialsim.visualizations',
]

setup(name='socialsim',
      version='0.1.2',
      packages=package_structure,
      license='License to be determined at a future date',
      url='URL to be determined at a future date',
      long_description='None',
      maintainer='Zachary New',
      maintainer_email='zachary.new@pnnl.gov',
      install_requires=[
            'pandas',
            'scipy>=1.2.1',
            'scikit-learn>=0.20.2',
            'fastdtw>=0.2.0',
            'pysal>=1.14.4',
            'tqdm>=4.31.1',
            'burst_detection>=0.1.3',
            'tsfresh>=0.11.2'
      ]
      )