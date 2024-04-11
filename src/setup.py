from setuptools import setup

setup(
    name="classify",
    version="0.1",
    description="Classify Presentations",
    author="ZeroSubstance",
    author_email="lynnpeter@proton.me",
    packages=["classify", "features_db"],
    install_requires=[],
    entry_points={
        "console_scripts": [
            "classify=classify.Main:main",
        ]
    },
)
