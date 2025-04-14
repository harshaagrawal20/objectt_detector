from setuptools import setup, find_packages

setup(
    name='object-detection-app',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A zero-shot object detection application using OpenCV and PyTorch.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'opencv-python',
        'torch',
        'torchvision',
        'numpy',
        'jsonschema'
    ],
    entry_points={
        'console_scripts': [
            'object-detection-app=main:main',
        ],
    },
)