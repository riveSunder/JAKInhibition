from setuptools import setup, find_packages


setup(\
        name="JAKInhibition", \
        version = "0.00", \
        install_requires =[\
                "numpy==1.23.1",\
                ],\
        packages=["jak"], \
        description = "estimating inhibition metrics of small molecules for JAK kinases"
    )


