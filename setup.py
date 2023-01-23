from setuptools import setup, find_packages

#  packages=["sepia", "testing"], \

setup(\
        name="JAKInhibition", \
        version = "0.00", \
        install_requires =[\
                "numpy==1.23.1",\
                ],\
        packages=find_packages(),\
        description = "estimating inhibition metrics of small molecules for JAK kinases"
    )


