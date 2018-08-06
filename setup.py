"""
Script for packaging.
"""
import setuptools

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="easy-gscv",
    version="0.1.0",
    author="Jasper Koops",
    author_email="jasperkoops91@gmail.com",
    description=(
        "A high level library gridsearch / cross evaluation library "
        "for scikit-learn"
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/Jasper-Koops/easy-gscv",
    packages=setuptools.find_packages(exclude=['venv']),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
    ],
    classifiers=(
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
