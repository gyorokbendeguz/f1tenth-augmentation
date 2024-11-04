from setuptools import setup, find_packages

setup(name='f1tenth_augmentation',
      version='1.0.0',
      description='Model augmentation for F1Tenth vehicle and MSD benchmark problem',
      author='Bendegúz Györök',
      author_email='gyorokbende@sztaki.hu',
      packages=find_packages(),
      install_requires=[
            "numpy==1.26.4",
            "torch",
            "matplotlib",
            "tqdm",
            "deepSI @ git+https://github.com/GerbenBeintema/deepSI@master"
        ]
      )