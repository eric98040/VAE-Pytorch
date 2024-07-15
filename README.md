# Variational AutoEncoder (VAE) Implementation

> Key points considered during the implementation of VAE are:


1) Error when converting from cuda tensor to numpy array: Convert numpy to torch.exp() or .pow() in the loss function and reparameterize function

2) When only returning x_hat from the Decoder, it is difficult to calculate the loss function, so return the output from the Encoder along with the intermediate steps

3) Even if the target does not have binary values of 0 or 1, but has values between [0, 1], BCE Loss can be used

</br>


If you want a further explanation about the paper, please refer to this [`link`](https://www.jaewon.work/paper-preview-vae-auto-encoding-variational-bayes-2/)



If you want the pytorch implementation using jupyter notebook, please refer to this [`link`](https://github.com/eric98040/VAE-Pytorch/blob/main/VAE.ipynb)



## Features

- VAE model implementation using PyTorch
- Training and evaluation scripts
- Utility functions for data processing and visualization

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/vae-project.git
    cd vae-project
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training the VAE

1. Run the training script:
    ```sh
    python train.py
    ```

### Evaluating the VAE

1. Run the evaluation script:
    ```sh
    python evaluate.py
    ```

### Project Structure

- `vae/`: Contains the VAE model and helper functions
    - `__init__.py`: Makes the `vae` directory a package
    - `model.py`: Defines the VAE model components (Encoder, Decoder, VAE)
    - `utils.py`: Utility functions for data processing and visualization
- `train.py`: Script for training the VAE
- `evaluate.py`: Script for evaluating the VAE
- `requirements.txt`: List of required packages
- `README.md`: Project documentation


```sh
vae-project/
│
├── vae/
│   ├── __init__.py
│   ├── model.py
│   └── utils.py
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md

```
  

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
