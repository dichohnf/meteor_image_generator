from setuptools import setup, find_packages

setup(
    name='taming-transformers',
    version='0.0.1',
    description='Taming Transformers for High-Resolution Image Synthesis',
    packages=find_packages(),
    install_requires=[
        'torch', 'numpy', 'tqdm', 'omegaconf', 'pillow', 'torchvision', 'pytorch-lightning',
        'torchvision', 'pytorch-lightning', 'pytorch-lightning', 'pytorch-lightning', 'einops',
        'requests', 'matplotlib'
    ],
)
