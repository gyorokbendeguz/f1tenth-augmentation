# f1tenth-augmentation
Scripts and model augmentation framework for identifying the dynamics of the F1Tenth vehicle, as well as an MSD example.

## Installation
First, clone the repository, then open the project folder as
```
$ git clone https://github.com/gyorokbendeguz/f1tenth-augmentation
$ cd f1tenth-augmentation/
```
It is recommended to use a virtual environment. On Linux/Bash run
```
$ python3 -m venv venv
$ source venv/bin/activate
```
On Windows run
```
$ python -m venv venv
$ ./venv/Scripts/activate
```
Then finally, install the package and its dependencies using pip
```
$ pip install -e .
```

## Standalone model augmentation toolbox
A standalone version of the `model_augmentation` toolbox used in this repository is available [here](https://github.com/AIMotionLab-SZTAKI/model-augmentation).

Or install directly as
```
$ pip install git+https://github.com/AIMotionLab-SZTAKI/model-augmentation@main
```

## Usage
To run a selected model augmentation structure on the F1Tenth identification task, just run one of the files from the `scripts` folder. E.g. the dynamic additive one as
```
$ python3 scripts/additive_dynamic_augm.py
```

The MSD example can be started as
```
$ python3 scripts/MSD_training.py
```

## License
See the [LICENSE](/LICENSE) file for license rights and limitations (MIT).
