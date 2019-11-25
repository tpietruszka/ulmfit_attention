from setuptools import setup, find_packages

setup(name="ulmfit_attention",
      version="0.0.1",
      author="Tomasz Pietruszka",
      author_email="tomek.pietruszka@gmail.com",
      description="Modules implementing improved pooling for ULMFiT text classification",
      packages=find_packages(),
      python_requires='>=3.6',
      install_requires=[
          'fastai>=1.0.59',
      ],
      tests_require=[
            'pytest',
      ],
      )
