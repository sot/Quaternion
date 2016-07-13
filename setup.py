import sys

from setuptools import setup
from setuptools.command.test import test as TestCommand
from Quaternion import __version__


class PyTest(TestCommand):
    user_options = [('args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.args = []

    def run_tests(self):
        # Import here because outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.args)
        sys.exit(errno)


setup(name='Quaternion',
      author='Jean Connelly',
      description='Quaternion object manipulation',
      author_email='jconnelly@cfa.harvard.edu',
      packages=['Quaternion', 'Quaternion.tests'],
      license=("New BSD/3-clause BSD License\nCopyright (c) 2016"
               " Smithsonian Astrophysical Observatory\nAll rights reserved."),
      download_url='http://pypi.python.org/pypi/Quaternion/',
      url='http://cxc.harvard.edu/mta/ASPECT/tool_doc/pydocs/Quaternion.html',
      version=__version__,
      zip_safe=False,
      tests_require=['pytest'],
      cmdclass={'test': PyTest},
      )
