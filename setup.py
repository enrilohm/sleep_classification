from setuptools import setup, find_packages
from os import path
import re
from os import listdir
from os.path import isfile, join

description="sleep_classification package"
here = path.abspath(path.dirname(__file__))

# package_name = re.sub(r".*/", "", here)
package_name = "sleep_classification"

with open(path.join(here, "requirements.txt"),"r") as f:
    requirements=f.read().splitlines()

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=package_name,
    version='0.1.0',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=f"https://github.com/enrilohm/{package_name}",
    author='Enrico Lohmann',
    author_email='enrico.lohmann@protonmail.com',
    python_requires='>=3.9',
    install_requires=requirements,
    packages=["sleep_classification"],
    package_data={
        package_name: [
            "data/*"
        ]
    },
)
