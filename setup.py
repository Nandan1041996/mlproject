"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from typing import List

HyPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    """This function will return list of requirements"""

    requirements =[]

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

        requirements = [req.replace('\n','') for req in requirements]

        if HyPEN_E_DOT in requirements:
            requirements.remove(HyPEN_E_DOT)

    return requirements




setup(
    name="mlproject",  # Required
    version="0.0.1",  # Required
    description="A sample Python project",  # Optional
    author="Mayur Nandanwar",  # Optional
    author_email="mayurnandanwar@ghcl.co.in",  # Optional
    packages = find_packages(),
    install_requires= get_requirements("requirements.txt")
)