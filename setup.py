import io
from setuptools import setup

with io.open('./README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='uctopic',
    packages=['uctopic'],
    version='0.2',
    license='MIT',
    description='A phrase embedding and topic mining (unsupervised aspect extraction) tool based on UCTopic',
    author='Jiacheng Li, Jingbo Shang, Julian McAuley',
    author_email='j9li@eng.ucsd.edu',
    url='https://github.com/JiachengLi1995/UCTopic',
    keywords=['phrase', 'embedding', 'uctopic', 'nlp'],
    install_requires=[
        "tqdm",
        "scikit-learn",
        "transformers==4.7.0",
        "torch>=1.7.0",
        "numpy>=1.17",
        "setuptools",
        "spacy"
    ]
)