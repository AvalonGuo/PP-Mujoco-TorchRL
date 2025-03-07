from setuptools import setup, find_packages

setup(
    name='ppmt',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'mujoco',
        'dm_control',
        'opencv-python',
        'torch',
        'torchvision',
        'torchaudio',
        'numpy',
    ],
)