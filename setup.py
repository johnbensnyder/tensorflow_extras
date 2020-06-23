import setuptools

setuptools.setup(
    name="tensorflow_extras",
    version="0.0.1.1",
    author="Ben Snyder",
    author_email="johnbensnyder@gmail.com",
    description="WIP addons for Tensorflow",
    long_description_content_type="text/markdown",
    url="https://github.com/johnbensnyder/tensorflow_extras",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "tensorflow >= 2.0",
    ]
)