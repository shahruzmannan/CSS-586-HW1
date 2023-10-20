# USPS-Handwritten-Data-Classification
I classified the USPS handwritten data, which consists of 9298 samples. I used a deep learning approach with Convolutional Neural Networks (CNNs) and
employed Grid Search with cross-validation to optimize the hyperparameters of the model.

# Data Loading and Preprocessing
* The data was loaded from a MATLAB file named `USPS_all.mat`.
* Each sample in the dataset was reshaped from a 1D vector of 256 elements to a 2D matrix of size 16x16 to be compatible with the CNN model.

# Model Creation and Hyperparameter Optimization
* A CNN model was used for this task. The model structure consisted of convolutional layers followed by
  * a max-pooling layer
  * a flattening layer
  * dense layers.
* The dataset was divided into 5 folds using `KFold`.
* `GridSearchCV` from scikit-learn was used to perform a grid search over various hyperparameters.
  * **Number of Layers**: `1`, `2`, and `3` conv layers.
  * **Number of Nodes**: `90`, `100`, `110`, and `120` nodes for the dense layer.
  * **Bias Initializer**: Both `zeros` and `ones` were tried as bias initializers.
  * **Activation Functions**: `relu` and `softmax` were tested.
  * **Optimizers**: `adam`, `sgd`, and `Adamax` optimizers were tested.
  * **Learning Rate**: `0.001, 0.01, 0.1, 0.2, and 0.3` learning rates were tested.
  * **Batch Size and Epochs**: Various combinations of batch sizes `(16, 32, 64, 128)` and epochs `(40, 50, 60, 70, 80)` were tried.
 
# Results
After performing the grid search, the best hyperparameters were found to be:

* Number of Layers: 1
* Number of Nodes: 90
* Bias Initializer: zeros
* Activation Function 1: relu
* Activation Function 2: softmax
* Optimizer: adam
* Number of Epochs: 70
* Batch Size: 16
* Learning Rate: 0.001

Using these hyperparameters, the model achieved an F1 score of approximately 0.9799.
