"""
This code sets the src/gaitml as a package for the Python project.
It uses setuptools to package the project, specifying metadata such as the version, author, description, 
and long description from a README file.
It also defines the package directory and finds all packages within the src directory.
"""


# importing necessary libraries
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "Gait-ML-Shap"
AUTHOR_USER_NAME = "UNB-Human-Performance-Laboratory"
SRC_REPO = "gaitml"
AUTHOR_EMAIL = "padhiar.karan@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="UNB Gait ML Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker" : f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"" : "src"},
    packages=setuptools.find_packages(where="src")
)

