# FaultDetection

## Introduction

This project introduces an intelligent fault detection method based on Deep Neural Networks (DNNs), specifically utilizing Stacked Autoencoders. The method employs four distinct datasets, encompassing diverse samples across varied operational conditions. The comprehensive dataset facilitates robust training and evaluation of the proposed fault detection framework.

## Methodology

The utilization of Sigmoid activation functions within the network architecture ensures efficient learning and inference processes. While the current implementation demonstrates satisfactory performance with a five-layer DNN configuration, further exploration into deeper network architectures remains a viable avenue for enhancement and experimentation.

## Instructions

To run this code, you need to ensure the following:

- **Python Environment**: Make sure you have Python installed on your system.

- **Dependencies**: Install the required libraries if you haven't already. You can do this using pip:

    ```
    pip install numpy scipy torch scikit-learn matplotlib
    ```

- **Data Directory Structure**: Ensure that your data directory structure changes based on the directory you save your data.

- **MAT Files**: Make sure you have the .mat files in the correct directories as specified in the code.

- **Run the Code**: You can run this code in any Python environment. Just execute it using a Python interpreter or in a Python script.

After running the code, you should have a file named `data_all.npy` containing the processed data. You can then proceed with further analysis or tasks using this data. To train each model, you just need to run the related file.

