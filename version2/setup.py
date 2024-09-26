import setuptools
from setuptools import setup

setup(name= "llm",
      version = 2.0,
      author = "25",
      license = "Apache License 2.0",
      package_dir={"":"version_temp"},
      packages = setuptools.find_namespace_packages(where="./version_temp"))