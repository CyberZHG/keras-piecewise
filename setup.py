from setuptools import setup

setup(
    name='keras-piecewise',
    version='0.2',
    packages=['keras_piecewise'],
    url='https://github.com/CyberZHG/keras-piecewise',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='A wrapper layer for splitting and accumulating sequential data',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy',
        'keras',
    ],
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
