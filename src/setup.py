from setuptools import setup

import ml

setup(
    name="ml",
    version=ml.__version__,
    description="Make Kaggle Great Again",
    author="Gianluca Rossi",
    author_email="gr.gianlucarossi@gmail.com",
    license="MIT",
    install_requires=[],  # TODO: fix this before publishing
    packages=["ml"],
    package_dir={"ml": "ml"},
)
