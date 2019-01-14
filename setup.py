from distutils.core import setup

package_structure = ['socialsim',
                     'socialsim.measurements',
                     'socialsim.measurements.cascade',
                     'socialsim.measurements.cross_platform',
                     'socialsim.measurements.group_formation',
                     'socialsim.measurements.infospread',
                     'socialsim.measurements.network',
                     'socialsim.metrics',
                     'socialsim.visualizations',
]

setup(name='socialsim',
      version='0.0.1',
      packages=package_structure,
      license='License to be determined at a future date',
      url='URL to be determined at a future date',
      long_description='None',
      maintainer='Zachary New',
      maintainer_email='zachary.new@pnnl.gov',
      )
