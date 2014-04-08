from setuptools import setup

setup(name='powergama',
      version='0.4.0',
      description='PowerGAMA - Power Grid And Market Analysis tool',
      url='https://bitbucket.org/harald_g_svendsen/powergama',
      author='Harald G Svendsen',
      author_email='harald.svendsen@sintef.no',
      license='MIT License (http://opensource.org/licenses/MIT)',
      packages=['powergama'],
	  data_files = [('', ['licence.txt'])],
      zip_safe = True
	 )
