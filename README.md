# Variational AutoEncoder (VAE) Implementation

> VAE를 구현하면서 중요하게 생각한 point들은 다음과 같다.
>
1) cuda tensor에서 numpy array 변환 시 오류: loss function, reparameterize 함수 안의 numpy를 torch.exp() or .pow() 등으로 변환
2) Decoder output을 x_hat만 반환할 경우 loss function 계산시 어려우므로 중간 단계인 Encoder에서 나온 output까지 같이 반환
3) 0 또는 1의 binary value를 target이 가지고 있는 binary classification 문제가 아니어도 [0, 1] 사이의 value를 가질 경우 BCE Loss 사용이 가능

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
