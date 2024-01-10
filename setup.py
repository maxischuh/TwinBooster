from setuptools import setup, find_packages


setup(
    name="twinbooster",
    version="0.1",
    packages=find_packages(
        include=[
            "twinbooster",
            # "twinbooster.*
        ]
    ),
    description="TwinBooster: Synergising Large Language Models with Barlow Twins and Gradient Boosting for Enhanced Molecular Property Prediction",
    author="Maximilian G. Schuh",
    install_requires=[
        "numpy",
        "pandas",
        "torch==2.0.1",
        "transformers==4.30.2",
        "datasets",
        "tqdm",
        "scikit-learn",
        "scipy",
        "joblib",
        "matplotlib",
        "lightgbm==3.3.5",
        "rdkit==2023.3.2",
        "pynvml",
        "ConfigSpace",
        "smac",
        "optuna",
        "jupyterlab",
        "pathlib",
    ],
    python_requires=">=3.8",
)
