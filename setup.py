from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."


def get_requirements(requirements_path: str) -> List[str]:
    with open(requirements_path) as file:
        requirements = file.read().split("\n")
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    return [r for r in requirements if r.strip()]


setup(
    name="Sales Forecasting",
    version="1.0.0",
    author="Ankit Kumar",
    author_email="ankit.aiml625@gmail.com",
    description="End-to-end ML pipeline for multivariate time series store sales forecasting.",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=get_requirements("requirements.txt"),
)