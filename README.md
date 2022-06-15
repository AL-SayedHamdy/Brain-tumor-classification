# Brain tumor classification
It's a clothes category classifier.

## General info
In this project, we will classify the human brain tumor. we will classify if there's a tumor or not. It's a simple binary classification deep learning model.

## Dataset
https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection

## Technologies
Project is created with:
* Convolution neural network
* TensorFlow version: 2.1.0
* Keras version: 2.2.4-tf
* scikit-learn version: '1.0.2'
* OpenCV
	
## CNN Structures for the brain tumor binary classification

----------------------------------------------------------------
    Layer (type)                 Output Shape              Param #   

    conv2d_1 (Conv2D)            (None, 128, 128, 32)      416       

    conv2d_2 (Conv2D)            (None, 128, 128, 32)      4128      

    batch_normalization_1 (Batch (None, 128, 128, 32)      128       

    max_pooling2d_1 (MaxPooling2 (None, 64, 64, 32)        0         

    dropout_1 (Dropout)          (None, 64, 64, 32)        0         

    conv2d_3 (Conv2D)            (None, 64, 64, 64)        8256      

    conv2d_4 (Conv2D)            (None, 64, 64, 64)        16448     

    batch_normalization_2 (Batch (None, 64, 64, 64)        256       

    max_pooling2d_2 (MaxPooling2 (None, 32, 32, 64)        0         

    dropout_2 (Dropout)          (None, 32, 32, 64)        0         

    flatten_1 (Flatten)          (None, 65536)             0         

    dense_1 (Dense)              (None, 512)               33554944  

    dropout_3 (Dropout)          (None, 512)               0         

    dense_2 (Dense)              (None, 2)                 1026     
