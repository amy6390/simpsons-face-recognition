# Getting started

## 1. Installing the repository
In order to get this repositoy on your local machine, please run the following command:
```
git clone https://github.com/amy6390/simpsons-face-recognition.git
```

## 2. Downloading all requirements
Open the project's root directory (`simpsons-face-recognition`) and then run the following command in your terminal:
```
pip install requirements.txt
```

## 3. Downloading training and testing data
All training and testing data has already been stored in the `data/external` directory. If you wish to download the data yourself, run the following command in your terminal:
```
python -m modeling.dataset
```

## 4. Training the model
An EfficientNetB0 model finetuned on this dataset over 25 epochs is already available in the `models` directory. If you wish to train an EfficientNetB0 model with a different number of epochs, please run 
```
python -m simpsons_face_recognition.modeling.train --epochs {number of epochs}
```

## 5. Prediction
To predict on the default test dataset, run the following command:
```
python -m simpsons_face_recognition.modeling.predict --model_path {path to your finetuned model}
```
To predict on your own custom dataset, run the following command:
```
python -m simpsons_face_recognition.modeling.predict --model_path {path to your finetuned model} --predictions_path {path to your dataset}
```
All predictions will be saved in `data/external/test_predictions.csv`. Every line in the csv is formatted as so: `{file name},{predicted class name}`.
