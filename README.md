# diffusion-model-cpp<br><sub>A C++ implementation of a diffusion model using libtorch</sub>

![Teaser image](./assets/sampled_images.png)

## Features
- Diffusion model implemented using libtorch
- Support for training and ~~inference~~
- Customizable parameters for training and model configuration
- Pure C++ implementation

## Tested environments
- C++17
- GCC 11.4.0
- CMake 3.30
- libtorch 2.1.0
- OpenCV 4.7.0 (for image processing)
- CUDAToolkit 12.3

## Getting Started

### Build
1. Clone the repository:
    ```sh
    git clone --recursive https://github.com/your_username/diffusion-model-cpp.git
    cd diffusion-model-cpp
    ```

2. Install dependencies:
    ```sh
    sudo apt-get update
    sudo apt-get install libopencv-dev
    ```

3. Download and extract libtorch:
    ```sh
    wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip
    unzip libtorch-cxx11-abi-shared-with-deps-2.3.1+cu121.zip
    ```

4. Build the project using CMake:
    ```sh
    mkdir build
    cd build
    cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
    make
    ```

### Training
1. Prepare your dataset and update the dataset path and log dir path in the configuration file `configs/sample.json`.

2. Run the training program:
    ```sh
    ./build/src/train configs/sample.json
    ```

3. The model and training logs will be saved in the log directory.

## Acknowledgements
- This project uses [libtorch](https://pytorch.org/cppdocs/) for implementing the diffusion model.
- OpenCV is used for image processing tasks.
- Inspiration and algorithms are based on recent research in the field of [diffusion models](https://github.com/crowsonkb/k-diffusion).
