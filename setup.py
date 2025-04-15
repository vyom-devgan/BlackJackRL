from setuptools import setup, find_packages

setup(
    name="deep",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26',
        'torch>=2.2',
        'matplotlib>=3.8',
        'pygame>=2.5',
        'tqdm>=4.66'
    ],
    entry_points={
        'console_scripts': [
            'blackjack-exp=deep.experiment:main'
        ]
    },
)
