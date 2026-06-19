from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    # Function will return the list of requirements
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines() # reads Numpy\n Pandas\n Seaborn\n
        requirements = [requirement.replace("\n", "") for requirement in requirements] # removes \n from the end of each requirement

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements


setup(
    name = "ML_Project",
    version = "0.0.1",
    author = "Aftabalam Makandar",
    author_email = "makandaraftabalam@gmail.com",
    packages = find_packages(),
    
    install_requires = get_requirements("requirements.txt")
)

## find_packages() -> Just go and see in how many folders we have __init__.py file and then consider those folders as packages.

"""

What does setup.py do?

It tells Python:

The package name
Version
Author
Dependencies
Which folders should be treated as packages

"""