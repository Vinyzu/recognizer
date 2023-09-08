import io
import os
from setuptools import setup, find_packages

import recaptcha_challenger

def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("botright", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    with io.open(os.path.join(os.path.dirname(__file__), *paths), encoding=kwargs.get("encoding", "utf8")) as open_file:
        content = open_file.read().strip()
    return content

# python setup.py sdist bdist_wheel && python -m twine upload dist/*
setup(
    name="recaptcha_challenger",
    version=recaptcha_challenger.__version__,
    keywords=["recaptcha", "recaptcha_challenger", "recaptcha-solver"],
    author="Vinyzu",
    maintainer="Vinyzu, QIN2DIM",
    maintainer_email="yaoqinse@gmail.com, bj.yan.pa@qq.com",
    url="https://github.com/Vinyzu/recaptcha-challenger",
    description="🦉 Gracefully face reCAPTCHA challenge with ModelHub embedded solution.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="GNU General Public License v3.0",
    packages=find_packages(include=["recaptcha_challenger", "recaptcha_challenger.*", "LICENSE"], exclude=["tests", ".github"]),
    install_requires=[
        "loguru>=0.6.0",
        "playwright>=1.26.1",
        "pydub>=0.25.1",
        "SpeechRecognition==3.8.1",
        "requests>=2.28.1",
        "opencv-python==4.5.5.64",
        "numpy>=1.22.4",
        "pyyaml~=6.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
)