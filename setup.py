from distutils.core import setup

setup(
    name='maskmaker',
    version='0.1.0',
    packages=['maskmaker'],
    url='https://github.com/nawendt/maskmaker',
    license='BSD 2-Clause License',
    author_email='nawendt@ou.edu',
    author='Nathan Wendt',
    description='Create masks for grids using shapefiles',
    dependency_links=['https://github.com/pmarshwx/pygridder/tarball/master'],
    install_requires=['pygridder', 'fiona', 'shapely']
)
