# Model Overview
The model is a Convolutional Neural Network (CNN) designed for classifying blurred images, specifically targeting camera shake artifacts. The architecture comprises:

- **3 Convolutional Layers**: Each layer uses ReLU activation and max pooling. The layers progressively increase feature depth from 32 to 128 channels.
- **3 Fully Connected Layers**: The first two layers use dropout for regularization and ReLU activation, with the final layer outputting 10 possible classes.
- **Dropout**: Applied in the first fully connected layer to prevent overfitting.

## Performance
- **Accuracy**: 99.11% on the test set, indicating strong generalization and minimal overfitting.
- **Training Loss**: Gradual reduction in loss to 0.006646, showing effective learning over epochs.
- **Test Loss**: Averaging 0.0394 at the final epoch, confirming good test set performance.


## Data Preprocessing
- **Dataset**: The dataset used is the n-MNIST dataset, which is publicly available online. Due to the large file size, the dataset is **not included in this repository**.
- **Transformation**: The dataset is converted to PyTorch tensors for further processing in the network.

## Project Context
This project was part of a series of experiments where the objective was to enhance an initial CNN code by testing various hyperparameters and modifications. Some key experiments included:

- **Number of Epochs**: Studying the effect of varying epochs on model accuracy and convergence.
- **Convolution Layers**: Adding additional convolution layers to observe changes in model performance.
- **Fully Connected Layers**: Experimenting with the number of fully connected layers and their impact on classification accuracy.
- **Feature Maps & Stride**: Exploring the effects of adjusting feature maps and stride values in convolutional layers to improve feature extraction.

The initial code and dataset were provided, and the objective was to optimize the architecture and hyperparameters to achieve better results.

## Training & Testing
- **Training**: The model is trained for a variable number of epochs using Adam optimizer and negative log likelihood loss. The training loop processes batches of 64 samples.
- **Testing**: Evaluation on the test set computes the average loss and accuracy.

## Code Overview
- `Net`: Defines the CNN model, including layers and forward pass.
- `nMNIST_Dataset`: Custom dataset class to load and preprocess the n-MNIST dataset.
- **Training & Testing Functions**: Separate functions to handle training and testing steps.

## Environment
- **Python**: Version 3.x
- **Libraries**: PyTorch, NumPy, Matplotlib (for visualization)
-**Intel i7 CPU Performance**: When running on an Intel i7 CPU, training takes approximately **3.5 minutes per epoch** and around **50 minutes** for the full training process. GPU will be much better for this application.
