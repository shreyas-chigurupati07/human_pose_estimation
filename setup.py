import setuptools
import os



with open("README.md", "r") as fh:
    long_description = fh.read()




__version__ = '0.0.1'



REPO_NAME = 'human_pose_estimation'
AUTHOR_USERNAME = 'shreyas-chigurupati07'
SRC_REPO = 'cnnEstimation'
AUTHOR_EMAIL = 'chigurupatishreyas@gmail.com'


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USERNAME,
    author_email=AUTHOR_EMAIL,
    description="A small package to estimate human pose using CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USERNAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USERNAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),

)