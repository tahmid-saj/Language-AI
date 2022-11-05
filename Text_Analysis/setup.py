import setuptools

long_description = "Text_Analysis"

with open("requirements.txt", "r") as requirements_file:
    external_packages = requirements_file.read()

setuptools.setup(
    name="Text_Analysis",
    version="0.0.1",
    author="tahmid-saj",
    description="Repo containing NLP projects ranging from text summarization, sentiment analysis and classification analysis using different sequence based models in TensorFlow and utilizing AWS for cloud computing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires = external_packages,
    package_dir={"":"src"},
    packages=setuptools.find_namespace_packages(where="src\\"),
    include_package_data=True,
    python_requires=">=3.7",
)