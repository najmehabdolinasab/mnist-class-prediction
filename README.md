# mnist-class-prediction
A Python script to build and evaluate neural network models for class prediction on the MNIST dataset. The models are trained using custom and dense layers, and their performance is compared based on accuracy and loss metrics.
# Neural Network Models for MNIST Dataset

This script is designed to build and evaluate two different neural network models using the MNIST dataset. The models are trained and tested using TensorFlow and Keras, and their performance is compared based on accuracy and loss.

## Project Description

In this project, the following steps are applied:

1. **Custom Layer Definition**:
   - A custom layer is defined using the `tf.keras.layers.Layer` class.
   - Weights (v, w, and b) are initialized with random normal values using the `add_weight` method.

2. **Dataset Preparation**:
   - The MNIST dataset is loaded, and the 28x28 pixel images are flattened into a 1D array.
   - The pixel values are normalized to the range [0, 1].
   - Labels are one-hot encoded using `tf.keras.utils.to_categorical` for binary classification.

3. **Model Definition**:
   - **Model 1 (Custom Layer)**: This model uses the custom layer defined earlier, followed by a ReLU activation function. The output layer has 3 units with a softmax activation.
   - **Model 2 (Dense Layers)**: This model uses standard dense layers with ReLU activation, with two layers of 128 and 64 units, and a softmax activation function for the output layer.

4. **Model Compilation**:
   - Both models are compiled using the Adam optimizer and categorical cross-entropy loss function. Accuracy is used as the evaluation metric.

5. **Model Training**:
   - Both models are trained for 50 epochs using the training data (`x_train`, `y_train`) and validated with the test data (`x_test`, `y_test`).
   - The training process is displayed using the `fit` function, showing loss and accuracy for both the training and test datasets.

6. **Visualization of Results**:
   - A comparison of the models' loss during training is plotted.
   - A comparison of the models' accuracy during training is also displayed to evaluate the performance of each model.

The script aims to compare which model (custom vs. Dense) performs better on the MNIST dataset.

## Libraries Used
- TensorFlow
- Keras
- NumPy
- Matplotlib

## How to Use
1. **Clone the repository**:
2. **Install required libraries**:
3. **Run the code**:
Open the `project_3.ipynb` file in Jupyter Notebook and follow the steps to build, train, and evaluate the neural network models.

## Contributing
If you would like to contribute to this project, please feel free to submit a pull request. All suggestions and improvements are welcome!
