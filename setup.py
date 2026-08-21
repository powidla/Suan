from setuptools import setup

setup(
    name="siu",
    packages=[
        'data',
        'inference',
        'train',
        'evaluation',
        'utils'
    ],
    package_dir={
        'data': './data',
        'inference': './inference',
        'train': './train',
        'evaluation': './evaluation',
        'utils': './utils'
    },
)
