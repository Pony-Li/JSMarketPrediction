from setuptools import setup, find_packages

setup(
    name='jsmarketprediction',
    version='0.1.0',
    author='Junyi Li',
    author_email='',
    description='Jane Street Real-Time Market Data Forecasting with GRU model + Online Learning',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Pony-Li/JSMarketPrediction',
    packages=find_packages(where='src') + ['kaggle_evaluation'],
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'polars',
        'tqdm',
        'protobuf',
        'grpcio',
    ],
    python_requires='>=3.9',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
