import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chromatics",
    packages=["chromatics"],
    version="0.1.1",
    author="Corin Wagen",
    author_email="corin.wagen@gmail.com",
    license="GPL 3.0",
    description="simple analysis of chromatographic data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/corinwagen/chromatics",
#    packages=setuptools.find_packages(),
    install_requires=["numpy", "lmfit", "scipy", "sklearn"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
#        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.7',
)
