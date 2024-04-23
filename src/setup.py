from setuptools import setup

setup(
    name="classify",
    version="0.3",
    description="Classify Presentations",
    author="ZeroSubstance",
    author_email="lynnpeter@proton.me",
    packages=["classify", "features_db", "pgdb"],
    install_requires=[],
    entry_points={
        "console_scripts": [
            "classify=classify.Main:main",
        ]
    },
)
