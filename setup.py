# setup.py

from setuptools import setup, find_packages

setup(
    name='vaila',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.4',
        'pandas==2.2.2',
        'matplotlib==3.8.4',
        'scipy==1.13.1',
        'dash==2.17.1',
        'ffmpeg-python==0.2.0',
        'mediapipe==0.10.14',
        'opencv-contrib-python==4.10.0.84',
        'imufusion==1.2.5',
    ],
    author='Guilherme Cesar, Ligia Mochida, Bruno Bedo, Paulo Santiago',
    author_email='paulosantiago@usp.br',
    description='Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox.',
    url='https://github.com/paulopreto/vaila-multimodaltoolbox',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11.8',
)
