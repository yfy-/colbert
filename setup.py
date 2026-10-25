import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

# Sources of the C++/CUDA extensions, which are compiled when first used.
package_data = {
    '': ['*.cpp', '*.cu'],
}

setuptools.setup(
    name='colbert',
    version='0.2.0',
    author='Omar Khattab',
    author_email='okhattab@stanford.edu',
    description="Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/stanford-futuredata/ColBERT',
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
    install_requires=[
        "bitarray",
        "GitPython",
        "ninja",
        "scipy",
        "tqdm",
        "transformers",
        "ujson"
    ],
    extras_require={
        'faiss-gpu': ['faiss-gpu>=1.7.0'],
        'torch': ['torch==1.13.1'],
        # Building a Sparcent index also needs the PISA tools and ciff's pisa2ciff.
        'sparcent': [
            'splade @ git+ssh://git@github.com/yfy-/splade.git@807887a99427c8cded4f07d9b9fd081d91a156e6',
            'bmp @ git+ssh://git@github.com/yfy-/BMP.git@f4df7f4c8b5ed646b4bb12f51b62111198f23e51#subdirectory=python',
        ],
        # The gdmis-rs static pruning method.
        'gdmis': ['gdmis @ git+ssh://git@codeberg.org/yfy/gdmis.git@763274f2e584ca5203347ba70ebeda8f33ef3da3'],
    },
    include_package_data=True,
    package_data=package_data,
)
