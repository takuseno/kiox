import os

from setuptools import setup, find_packages

# get __version__ variable
here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, 'kiox', '_version.py')).read())

if __name__ == "__main__":
    setup(name="kiox",
          version=__version__,
          description="A composable experience replay buffer library",
          long_description=open("README.md").read(),
          long_description_content_type="text/markdown",
          url="https://github.com/takuseno/kiox",
          author="Takuma Seno",
          author_email="takuma.seno@gmail.com",
          license="MIT License",
          classifiers=["Development Status :: 4 - Beta",
                       "Intended Audience :: Developers",
                       "Intended Audience :: Education",
                       "Intended Audience :: Science/Research",
                       "Topic :: Scientific/Engineering",
                       "Topic :: Scientific/Engineering :: Artificial Intelligence",
                       "Programming Language :: Python :: 3.7",
                       "Programming Language :: Python :: 3.8",
                       "Programming Language :: Python :: Implementation :: CPython",
                       "Operating System :: POSIX :: Linux",
                       'Operating System :: Microsoft :: Windows',
                       "Operating System :: MacOS :: MacOS X"],
          install_requires=["numpy",
                            "h5py",
                            "typing-extensions",
                            "grpcio"],
          packages=find_packages(exclude=["tests*"]),
          python_requires=">=3.7.0",
          zip_safe=True)
