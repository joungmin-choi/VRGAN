from setuptools import setup, find_packages

setup(
    name='vrgan',
    version='0.1.0',
    description='VRGAN: a deep generative framework for unsupervised anomaly detection in multivariate time-series data',
    author='Joung Min Choi',
    author_email='joungmin@vt.edu',
    url='https://github.com/joungmin-choi/VRGAN',
    py_modules=['VRGAN'],
    install_requires=['torch', 'pandas', 'numpy', 'scikit-learn'],
    packages=find_packages(exclude=[]),
    keywords=['vrgan', 'anomaly-detection', 'time-series'],
    python_requires='>=3.7',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
