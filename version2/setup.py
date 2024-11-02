import setuptools
from setuptools import setup

# Setup function to specify the package details
setup(
    name="llm",  # Name of the package
    version=2.0,  # Version of the package
    author="25",  # Author of the package
    license="Apache License 2.0",  # License type
    package_dir={"": "version_temp"},  # Directory containing the package
    packages=setuptools.find_namespace_packages(where="./version_temp")  # Find and include all packages in the specified directory
)