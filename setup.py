from setuptools import setup, find_packages
HYPEN_E_DOT = "-e ."
def get_requirements(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        requirements = [line.strip() for line in lines if line.strip()]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        return requirements

setup(
    name="AWS,AZURE_DEPLOYMENT",
    version="0.0.1",
    author="Amaresh",
    author_email="amareshbharanagar@gmail.com",
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')
)