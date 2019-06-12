## Grab Computer Vision Challenge-Cars Classification
Car classifier is built using transfer learning technique. By fine tuning the pretrained CNN - Inception-ResnetV2, the model performance is able to reach the maximum accuracy **83.40%**, F1 score **83.40%** and error rate **0.677** on test set.

### Model Implementation
Model training is done and demonstrated in [car_classifier.ipynb](https://github.com/vincenttang96/car-classification/blob/master/car_classifier.ipynb). Data acquire, preprocessing, model implementation and training results could be found under this notebook.

### Load Trained .h5 Model
Trained classification model is saved in h5 format and listed here [here](https://github.com/vincenttang96/car-classification/tree/master/trained_model). In the demo below, **stage2_model_weights.h5** is loaded for classification model. More details about the code please refer to [predict.ipynb](https://github.com/vincenttang96/car-classification/blob/master/predict.ipynb).
```
model = load_model("trained_model/stage2_model_weights.h5")
```

### Model Testing
Model evaluation and testing could be done in [predict.ipynb](https://github.com/vincenttang96/car-classification/blob/master/predict.ipynb). Code section below shows the model prediction on testing dataset. Then, prediction result will be output as a .txt file. [Here (predictions.txt)](https://github.com/vincenttang96/car-classification/blob/master/predictions.txt) already shows the prediction output from the trained model. More details about the code please refer to the notebook.
```
predictions = predict(cars_test_annos, cars_meta, path_to_test_images, saved_model=model)
```
