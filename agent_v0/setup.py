import setuptools
from setuptools import setup

setup(name= "llm",
      verions = 1,
      author = "25",
      license = "Apache License 2.0",
      package_dir={"":"version1"},
      packages = setuptools.find_namespace_packages(where="./version1"))