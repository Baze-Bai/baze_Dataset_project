from setuptools import setup, find_packages

setup(
    name='ECM_fit',
    version='1.0',
    packages=find_packages(),
    py_modules=['ECM_func', 'main', 'test'],  
    install_requires=[
        'Pandas==2.1.1',
        'setuptools==75.6.0',
        'NumPy==1.26.0',
        'SciPy==1.11.3',
        'Matplotlib==3.8.0',
        'impedance==0.5.1'
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