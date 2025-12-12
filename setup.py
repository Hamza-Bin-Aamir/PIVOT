"""Setup configuration for PIVOT package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "readme.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

dev_requirements = (this_directory / "requirements-dev.txt").read_text().splitlines()
dev_requirements = [r.strip() for r in dev_requirements if r.strip() and not r.startswith("#")]

setup(
    name="pivot",
    version="0.1.0",
    description="Pulmonary Imaging for Volume Oncology Triage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="PIVOT Team",
    author_email="",
    url="https://github.com/Hamza-Bin-Aamir/PIVOT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="medical-imaging, deep-learning, lung-cancer, CT-scan, pytorch, monai",
    entry_points={
        "console_scripts": [
            "pivot-train=train.main:main",
            "pivot-infer=inference.main:main",
            "pivot-preprocess=data.preprocess:main",
        ],
    },
)
