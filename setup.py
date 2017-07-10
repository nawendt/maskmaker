from distutils.core import setup
from maskmaker import __author__, __email__, __version__

dependencies = ['fiona', 'numpy', 'scikit-image', 'scipy', 'shapely']

setup(
    name='maskmaker',
    version=__version__,
    packages=['maskmaker'],
    url='https://github.com/nawendt/maskmaker',
    license='BSD 2-Clause License',
    author_email=__email__,
    author=__author__,
    description='Create masks for grids using shapefiles',
    install_requires=dependencies
)
