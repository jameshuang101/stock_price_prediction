from distutils.core import setup

setup(
    name="stock_prediction",
    version="0.0.1",
    author="James Huang",
    author_email="jhuan44044@gmail.com",
    packages=[
        "app", "src"
    ],
    long_description=open("README.md").read(),
)
