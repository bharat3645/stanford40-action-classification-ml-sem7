"""
Setup script for Stanford 40 Actions Classification
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stanford40-action-classification",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep Learning System for Human Action Recognition using Stanford 40 Actions Dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stanford40_action_classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="action-recognition computer-vision deep-learning transfer-learning image-classification",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/stanford40_action_classification/issues",
        "Source": "https://github.com/yourusername/stanford40_action_classification",
        "Documentation": "https://github.com/yourusername/stanford40_action_classification/blob/main/README.md",
    },
)
