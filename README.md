# AIR-Act2Act

## Setting 
-   Python = 3.6.1     
-   Tensorflow = 1.11.0    

## Source files
    .
    ├── act2act
    │   ├── draw.py                 # Load test data and draw 3d plot
    │   ├── model.py                # LSTM-based behavior generation model
    │   └── train.py                # Train model
    ├── data
    │   ├── extracted files         # Training data extracted from AIR-Act2Act data 
    │   ├── joint files             # Joint files of AIR-Act2Act data
    │   ├── copy_files.py           # (not using)
    │   └── extract_data.py         # Extract training data from AIR-Act2Act data
    ├── model                       # Trained model and behavior generation results
    ├── utils
    │   ├── AIR.py                  # Load AIR-Act2Act data
    │   ├── nao.py                  # Convert between 3D joint position and NAO joint angles
    │   └── rnn_cell_extensions.py  # Define rnn cells with residual connection
    ├── constants.py                # Global constants
    ├── LICENSE.md
    ├── LICENSE_ko.md
    └── README.md

## Behavior generation model

![model](https://user-images.githubusercontent.com/13827622/62700274-779ae000-ba1c-11e9-8c4e-33ed33fce811.png)

## How to train the model

1.   Download AIR-Act2Act from [here](https://drive.google.com/file/d/1Z_ZECV9uqZgNrKCuvEuN4t-1_TXWMxio/view?usp=sharing) and put in 'data/joint files/'.  
2.   Run 'data/extract_data.py' to extract training data.   
3.   Run 'act2act/train.py' to train the model.

## LICENSE
This software is a part of AIR, and follows the [AIR License and Service Agreement](LICENSE.md).
