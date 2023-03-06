from setuptools import setup, find_packages


setup(
    name="MLOpsScaffold",
    version="1.0.0",
    url="https://github.com/gkamtzir/mlops-scaffold",
    author="Georgios Kamtziridis",
    author_email="georgekam96@gmail.com",
    description="Basic MLOps scaffold for data science projects.",
    packages=find_packages(),
    install_requires=["pandas", "numpy", "plotly", "matplotlib", "scikit-learn", "mlflow", "scikit-optimize"],
)