# Convolutional Neural Network (CNN) from Scratch using Numpy

## Overview

This project demonstrates the implementation of a Convolutional Neural Network (CNN) from scratch using only Numpy. The CNN is designed to classify handwritten digits from the MNIST dataset. This project is inspired by and builds upon the excellent guide by Victor Zhou.

## Features

- **Convolution Layer**: Implemented with a 3x3 filter using valid padding. Performs matrix multiplication (dot product) and includes the Sobel filter for edge detection.
- **Pooling Layer**: Max pooling with a size of 2, which reduces dimensionality by selecting the maximum value from each 2x2 region.
- **Softmax Function**: Used for multi-class classification with 10 nodes, corresponding to the 10 digits in the MNIST dataset.
- **Cross-Entropy Loss**: Employed for the loss function during training.
- **Backpropagation**: Implemented to update the weights based on the error gradients.

## Results

The CNN was trained for 3 epochs on the MNIST dataset, achieving an accuracy of 75-85%.

### Prerequisites

- Python 3.6 or higher
- Required Python libraries (listed in `requirements.txt`)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to Victor Zhou for his comprehensive guide on CNN implementation which served as a valuable resource for this project.
