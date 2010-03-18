from setuptools import setup
setup(name='Quaternion',
      author = 'Jean Connelly',
      description='Quaternion object manipulation',
      author_email = 'jconnelly@cfa.harvard.edu',
      py_modules = ['Quaternion'],
      license = "New BSD/3-clause BSD License\nCopyright (c) 2009 Smithsonian Astrophysical Observatory\nAll rights reserved.",
      download_url = 'http://pypi.python.org/pypi/Quaternion/',
      url = 'http://cxc.harvard.edu/mta/ASPECT/tool_doc/pydocs/Quaternion.html',
      scripts = [ 'test.py' ],
      version='0.03.1',
      zip_safe=False,
      package_data={}
      )
