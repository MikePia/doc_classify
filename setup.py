from setuptools import setup, find_packages

setup(
    name="classify",
    version="0.3.5",
    description="Classify Presentations",
    author="ZeroSubstance",
    author_email="lynnpeter@proton.me",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        "python-dotenv",
        "pandas",
        "joblib",
        "xgboost",
        "catboost",
        "tqdm",
        "spacy",
        "PyMuPDF",
        "pytesseract",
        "scikit-learn==1.4.1post1",
        
        ],
    entry_points={
        "console_scripts": [
            "classify=classify.Main:main",
        ]
    },
)
