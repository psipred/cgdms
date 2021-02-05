import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="cgdms",
    version="0.1",
    author="Joe G Greener",
    author_email="j.greener@ucl.ac.uk",
    description="Differentiable molecular simulation of proteins with a coarse-grained potential",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/psipred/cgdms",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="MIT",
    keywords="protein potential force-field coarse-grained automatic-differentiation",
    scripts=["bin/cgdms"],
    install_requires=["numpy", "biopython", "PeptideBuilder"],
    include_package_data=True,
)
