# simpsons-face-recognition

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A ML project to train an EfficientNetB0 model to classify Simpsons characters.

## Installation
Please run the following command to download this repository on your local machine:
```
git commit https://github.com/amy6390/simpsons-face-recognition.git
```
Please see the `docs` folder for documentation.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   └── external       <- Data from third party sources.
│
├── docs               <- Documentation for how to use this repository.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         simpsons-face-recognition and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── simpsons-face-recognition   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes simpsons-face-recognition a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
## Credits
Training and testing data comes from https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset/

