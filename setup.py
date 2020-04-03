import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nearestPD", # Replace with your own username
    version="0.0.1",
    author="John D’Errico and Ahmed Fasih",
    author_email="author@example.com",
    description="A python implementation for computing nearest positive definite matrix for a given matrix",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/esiivola/nearestPD",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)