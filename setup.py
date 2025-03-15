from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    """This function returns a list of requirements from a given file path."""
     
    requirements=[]
    HYPHEN_E_DOT='-e.'

    with open(file_path) as f:
        requiurements=f.readlines()
        requiurements=[req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        return requirements
    

setup(
    name='mlproject',
    version='0.0.1',
    author='harsha',
    author_email='harshamiisal@gmail.com',
    packages=find_packages,
    install_requires=get_requirements('requirements.txt')

)