from setuptools import setup

exec(open('powergama/version.py').read())

setup(name='powergama',
      version=__version__,
      description='PowerGAMA - Power Grid And Market Analysis tool',
	  long_description = 'PowerGAMA is a Python-based lightweight simulation tool for high level analyses of renewable energy integration in large power systems. The simulation tool optimises the generation dispatch, i.e. the power output from all generators in the power system, based on marginal costs for each timestep over a given period, for example one year. It takes into account the variable power available for solar, hydro and wind power generators. It also takes into account the variability of demand. Moreover, it is flow-based meaning that the power flow in the AC grid is determined by physical power flow equations.',
      url='https://bitbucket.org/harald_g_svendsen/powergama',
      author='Harald G Svendsen',
      author_email='harald.svendsen@sintef.no',
      license='MIT License (http://opensource.org/licenses/MIT)',
      packages=['powergama'],
      zip_safe = True,
	  classifiers = [
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 2.7',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.3',
		'Programming Language :: Python :: 3.4'
	  ],
	  keywords = 'power systems, grid integration, renewable energy',
	  install_requires = [
		'PuLP>=1.5.6'
		],
	 )
