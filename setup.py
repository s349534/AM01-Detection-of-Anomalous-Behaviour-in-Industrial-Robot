"""Setup configuration for AM01 anomaly detection package."""

from setuptools import setup, find_packages

setup(
    name="am01-anomaly-detection",
    version="0.1.0",
    author="Student Group",
    author_email="",
    description="Anomaly detection for industrial robots using adversarial autoencoders",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.10",
)