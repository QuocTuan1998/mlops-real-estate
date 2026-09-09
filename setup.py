from setuptools import setup, find_packages

setup(
    name='mlops-real-estate',
    version='0.1.0',
    author='Tuan Nguyen',
    author_email='tuan.nguyen@gmail.com',
    description='A MLOps project for Real Estate Price Prediction',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'PyYAML',
        'jupyter',
        'pytest',
        'mlflow',
        'fastapi',
        'uvicorn[standard]',
        'pydantic',
        'httpx'
    ],
    entry_points={
        'console_scripts': [
            'run_pipeline=src.pipeline.update_pipeline:main',
        ],
    },
)
