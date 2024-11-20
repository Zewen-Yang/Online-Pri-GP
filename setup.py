# Copyright (c) by Zewen Yang under GPL-3.0 license
# Last modified: Zewen Yang 02/2024

from setuptools import find_packages, setup
from typing import List
from pathlib import Path

# Constants
PROJECT_NAME = "Online-Pri-GP"
VERSION = "0.0.1"
AUTHOR = "Zewen-Yang"
AUTHOR_EMAIL = "zewenreal@gmail.com"
DESCRIPTION = "Online Prior-Aware Gaussian Process"
REQUIREMENTS_FILE = "requirements.txt"

def get_requirements(file_path: str) -> List[str]:
    """
    Read and parse requirements from requirements.txt file.
    
    Args:
        file_path (str): Path to the requirements file
        
    Returns:
        List[str]: List of package requirements
    """
    requirements = []
    try:
        with Path(file_path).open() as file_obj:
            requirements = [
                req.strip() for req in file_obj.readlines()
                if req.strip() and not req.startswith('-e ')
            ]
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        return []
    
    return requirements

def setup_package():
    """Main setup configuration."""
    setup(
        name=PROJECT_NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=Path("README.md").read_text(encoding="utf-8")
        if Path("README.md").exists() else "",
        long_description_content_type="text/markdown",
        packages=find_packages(exclude=["tests*", "docs*"]),
        install_requires=get_requirements(REQUIREMENTS_FILE),
        python_requires=">=3.10,<3.12",
        classifiers=[
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Programming Language :: Python :: 3",
        ],
    )

if __name__ == "__main__":
    setup_package()