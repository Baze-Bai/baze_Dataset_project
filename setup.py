from setuptools import setup, find_packages

setup(
    name='ECM_fit',
    version='1.0',
    packages=find_packages(),
    py_modules=['ECM_func', 'main', 'test'],  
    install_requires=[
        'pandas',
        'setuptools',
        'chardet',
        'openpyxl',
        'xlrd',
        'numpy',
        'scipy',
        'matplotlib',
        'impedance',
        'unittest',
    ],
    entry_points={
        'console_scripts': [
            'ECM_fit=main:main',
        ],
    },
    author='Zejun Bai',
    author_email='zejun.bai@duke.edu',
    description='A tool to fit EIS data and get the ECM model.',
)