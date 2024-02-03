from setuptools import setup, find_packages

    
setup(
    name='ThesisAISourcce',
    version='1.0.0',
    author='Bas Blokzijl',
    description='This project contains the source code for the different AI models used in my thesis.'
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Bblokz/Thesis_SelfDriving_AI',
    packages=find_packages(),
    install_requires=[
        'matplotlib==3.8.0',
        'numpy==1.26.0',
        'pandas==2.1.1',
        'Pillow==10.1.0',
        'sympy==1.12',
        'tensorflow==2.14.0',
        # PyTorch installation command below is generic; see note below.
        'torch==2.1.0+cu121',
        'torchaudio==2.1.0',
        'torchvision==0.16.0+cu121',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11.0',
)

