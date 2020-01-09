# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

setup(name='Quaternion',
      author='Jean Connelly',
      description='Quaternion object manipulation',
      author_email='jconnelly@cfa.harvard.edu',
      packages=['Quaternion', 'Quaternion.tests'],
      license=("New BSD/3-clause BSD License\nCopyright (c) 2016"
               " Smithsonian Astrophysical Observatory\nAll rights reserved."),
      download_url='http://pypi.python.org/pypi/Quaternion/',
      url='http://cxc.harvard.edu/mta/ASPECT/tool_doc/pydocs/Quaternion.html',
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      zip_safe=False,
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
