"""
Created 07-01-21 by Mojtaba Heydari <mheydari@ur.rochester.edu>

"""


# Local imports
# None.

# Third party imports
# None.

# Python standard library imports
import setuptools
from setuptools import find_packages
import distutils.cmd


# Required packages
REQUIRED_PACKAGES = [
    'numpy>=1.21.0',
    'cython',
    'librosa>=0.8.0',
    'scipy',
    'mido>=1.2.6',
    'pytest',
    #'pyaudio',
    'madmom>=0.16.1',
    'torch>=1.9.0',
    'Matplotlib',
    'tensorboard>=2.0',
    'pyyaml>=5.0',
]


class MakeReqsCommand(distutils.cmd.Command):
  """A custom command to export requirements to a requirements.txt file."""

  description = 'Export requirements to a requirements.txt file.'
  user_options = []

  def initialize_options(self):
    """Set default values for options."""
    pass

  def finalize_options(self):
    """Post-process options."""
    pass

  def run(self):
    """Run command."""
    with open('./requirements.txt', 'w') as f:
        for req in REQUIRED_PACKAGES:
            f.write(req)
            f.write('\n')


setuptools.setup(
    cmdclass={
        'make_reqs': MakeReqsCommand
    },

    # Package details
    name="BeatNet",
    version="1.2.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,

    # Metadata to display on PyPI
    author="Mojtaba Heydari",
    author_email="mheydari@ur.rochester.edu",
    description="A package for Real-time and offline music beat, downbeat tempo and meter tracking using BeatNet AI, with training support",
    keywords="Beat tracking, Downbeat tracking, meter detection, tempo tracking, particle filtering, real-time beat, real-time tempo, music information retrieval, training",
    url="https://github.com/mjhydri/BeatNet"


    # CLI - not developed yet
    #entry_points = {
    #    'console_scripts': ['beatnet=beatnet.cli:main']
    #}
)
