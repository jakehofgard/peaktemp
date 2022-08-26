from setuptools import setup, find_packages

setup(
    name='peaktemp',
    version='0.1.1',
    license='MIT',
    author="Jake Hofgard",
    author_email='whofgard@stanford.edu',
    packages=find_packages('peaktemp'),
    package_dir={'': 'peaktemp'},
    url='https://github.com/jakehofgard/peaktemp',
    keywords='climate, temperature, forecasting, peak load',
    python_requires='>=3.8',
    install_requires=[
        'matplotlib',
        'pandas',
        'numpy',
        'seaborn',
        'cdsapi',
        'xarray',
        'xgboost',
        'scikit-learn',
        'datetime',
        'geopy'
    ],
)
