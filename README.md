# MAD-Unet

Object segmentation of near surface magnetic field dat based on deep convolutional neural networks


## Download

You can download the latest code and data from the following link:

[Download Code.zip](https://github.com/oliver0829/MAD-Unet/releases/download/CodeandData/Code.zip)

To download the code:
1. Click the link above to download `Code.zip`.
2. Extract the contents of `Code.zip`.

## Installation

Once you've downloaded and extracted the code, follow these steps to set up the project:

1. **Unzip the Code.zip file**:

    After downloading `Code.zip`, unzip it:

    ```bash
    cd Code
    ```

3. **Set up the environment (this does not include PyTorch)**:

    ```bash
    conda env create -f environment.yml
    conda activate mad-unet
    ```
4. **Installing PyTorch**:

   PyTorch installation requires selecting the right configuration based on your hardware (CUDA or CPU). Follow these steps:

   - If you are using a **GPU with CUDA support**, visit [PyTorch's official installation page](https://pytorch.org/get-started/locally/) and generate the correct installation command for your system. For example, to install with CUDA 11.7:

     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
     ```

   - If you are using **CPU only** (no GPU support), install PyTorch with:

     ```bash
     pip install torch torchvision torchaudio
     ```

## Usage

After setting up the data and the environment, run the following command to test the model:

```bash
python test_{}\getResult.py
```

## License

This project is licensed under the MIT License


## Contact

For any questions or suggestions, feel free to reach out:

- **Name**: Qiang Li
- **Email**: liqiangwcr@njust.edu.cn
