from setuptools import setup, find_packages

setup(
    name="ulmfit_attention",
    version="0.0.1",
    author="Tomasz Pietruszka",
    author_email="tomek.pietruszka@gmail.com",
    description="Modules implementing improved pooling for ULMFiT text classification",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "fastai>=1.0.59",
        "torch>=1.3.1",
        "hyperspace_explorer>=0.3.1",
    ],
    tests_require=["pytest",],
)
