## CNN-Kaggle-Pratice
*** 
### List

[Bees_Wasps_Insect_Other Step Description](https://github.com/ahoucbvtw/CNN-Kaggle-Pratice#bees_wasps_insect_other-step-description)

[HandGesture_Recognition Step Description](https://github.com/ahoucbvtw/CNN-Kaggle-Pratice#handgesture_recognition-step-description)

[Intel Step Description](https://github.com/ahoucbvtw/CNN-Kaggle-Pratice#intel-step-description)

*** 
### DataSet URL :

Bees_Wasps_Insect_Other Kaggle's DataSet : [Bee or wasp?](https://www.kaggle.com/jerzydziewierz/bee-vs-wasp)     
HandGesture_Recognition Kaggle's DataSet : [Hand Gesture Recognition Database](https://www.kaggle.com/gti-upm/leapgestrecog)     
Intel Kaggle's DataSet : [Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification)      

***
### Bees_Wasps_Insect_Other Step Description
1. Unzip dataset and load **labels.csv**

![ ](https://raw.githubusercontent.com/ahoucbvtw/CNN-Kaggle-Pratice/main/Bees_Wasps_Insect_Other/Picture/ReadCSV.jpg )

2. Use **labels.csv** to  make a **final_validation dataframe**, and this dataframe is used to make a **Confusion_Matrix** in the final.

![ ](https://raw.githubusercontent.com/ahoucbvtw/CNN-Kaggle-Pratice/main/Bees_Wasps_Insect_Other/Picture/Final_validationDF.jpg )

3. Use **labels.csv** to  make a **train_df**, and this dataframe is used for our training.

![ ](https://raw.githubusercontent.com/ahoucbvtw/CNN-Kaggle-Pratice/main/Bees_Wasps_Insect_Other/Picture/Train_DF.jpg )

4. Use **train_test_split** to separate x_train, x_test, y_train and y_test
```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_df["path"], train_df["answer"], test_size = 0.1, random_state = 5566)
```

5. Make the folders for Training and Final_Validation, and move the pictures to these folders according to **final_validation dataframe** and **train_df**

6. Use Generate to preprocess images.
```
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        path_Train,
        target_size = (224, 224),
        batch_size = 20,
        class_mode = 'categorical',
        )

test_generator = test_datagen.flow_from_directory(
        path_Test,
        target_size = (224, 224),
        batch_size = 20,
        class_mode = 'categorical',
        )
```

7. Build CNN NetWork(self)

![ ](https://raw.githubusercontent.com/ahoucbvtw/CNN-Kaggle-Pratice/main/Bees_Wasps_Insect_Other/Picture/Model-1.jpg)

![ ](https://raw.githubusercontent.com/ahoucbvtw/CNN-Kaggle-Pratice/main/Bees_Wasps_Insect_Other/Picture/Model-2.png)

8. Training
```
Epoch 1/300
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.2497s vs `on_train_batch_end` time: 0.5254s). Check your callbacks.
435/435 - 348s - loss: 1.1557 - accuracy: 0.5107 - val_loss: 1.0722 - val_accuracy: 0.5424
Epoch 2/300
435/435 - 341s - loss: 1.0299 - accuracy: 0.5760 - val_loss: 0.9980 - val_accuracy: 0.5787
Epoch 3/300
435/435 - 338s - loss: 0.9908 - accuracy: 0.5908 - val_loss: 1.0534 - val_accuracy: 0.5870
Epoch 4/300
435/435 - 343s - loss: 0.9239 - accuracy: 0.6101 - val_loss: 0.9252 - val_accuracy: 0.6242
Epoch 5/300
435/435 - 344s - loss: 0.8474 - accuracy: 0.6340 - val_loss: 0.8353 - val_accuracy: 0.6335
Epoch 6/300
435/435 - 344s - loss: 0.7915 - accuracy: 0.6802 - val_loss: 0.7494 - val_accuracy: 0.7174
Epoch 7/300
435/435 - 339s - loss: 0.7451 - accuracy: 0.7025 - val_loss: 0.7538 - val_accuracy: 0.7257
Epoch 8/300
435/435 - 344s - loss: 0.7151 - accuracy: 0.7203 - val_loss: 0.6834 - val_accuracy: 0.7391
Epoch 9/300
435/435 - 344s - loss: 0.6871 - accuracy: 0.7300 - val_loss: 0.6567 - val_accuracy: 0.7598
Epoch 10/300
435/435 - 343s - loss: 0.6498 - accuracy: 0.7482 - val_loss: 0.6296 - val_accuracy: 0.7484
Epoch 11/300
435/435 - 340s - loss: 0.6203 - accuracy: 0.7585 - val_loss: 0.6735 - val_accuracy: 0.7557
Epoch 12/300
435/435 - 344s - loss: 0.6074 - accuracy: 0.7623 - val_loss: 0.6110 - val_accuracy: 0.7671
Epoch 13/300
435/435 - 340s - loss: 0.5566 - accuracy: 0.7868 - val_loss: 0.6324 - val_accuracy: 0.7660
Epoch 14/300
435/435 - 345s - loss: 0.5597 - accuracy: 0.7834 - val_loss: 0.5917 - val_accuracy: 0.7826
Epoch 15/300
435/435 - 345s - loss: 0.5162 - accuracy: 0.8073 - val_loss: 0.5847 - val_accuracy: 0.7971
Epoch 16/300
435/435 - 344s - loss: 0.4711 - accuracy: 0.8283 - val_loss: 0.4670 - val_accuracy: 0.8209
Epoch 17/300
435/435 - 346s - loss: 0.4421 - accuracy: 0.8434 - val_loss: 0.4618 - val_accuracy: 0.8364
Epoch 18/300
435/435 - 342s - loss: 0.3985 - accuracy: 0.8573 - val_loss: 0.4944 - val_accuracy: 0.8240
Epoch 19/300
435/435 - 342s - loss: 0.3884 - accuracy: 0.8617 - val_loss: 0.4921 - val_accuracy: 0.8106
Epoch 20/300
435/435 - 342s - loss: 0.3644 - accuracy: 0.8692 - val_loss: 0.4785 - val_accuracy: 0.8323
Epoch 21/300
435/435 - 346s - loss: 0.3257 - accuracy: 0.8862 - val_loss: 0.4208 - val_accuracy: 0.8458
Epoch 22/300
435/435 - 342s - loss: 0.3025 - accuracy: 0.8963 - val_loss: 0.5149 - val_accuracy: 0.8292
Epoch 23/300
435/435 - 342s - loss: 0.2683 - accuracy: 0.9032 - val_loss: 0.4598 - val_accuracy: 0.8623
Epoch 24/300
435/435 - 343s - loss: 0.2575 - accuracy: 0.9084 - val_loss: 0.5889 - val_accuracy: 0.8157
Epoch 25/300
435/435 - 342s - loss: 0.2198 - accuracy: 0.9229 - val_loss: 0.4733 - val_accuracy: 0.8602
Epoch 26/300
435/435 - 342s - loss: 0.1993 - accuracy: 0.9303 - val_loss: 0.5471 - val_accuracy: 0.8354
Epoch 27/300
435/435 - 343s - loss: 0.1917 - accuracy: 0.9333 - val_loss: 0.4732 - val_accuracy: 0.8582
Epoch 28/300
435/435 - 343s - loss: 0.1508 - accuracy: 0.9488 - val_loss: 0.5301 - val_accuracy: 0.8530
Epoch 29/300
435/435 - 343s - loss: 0.1417 - accuracy: 0.9498 - val_loss: 0.6014 - val_accuracy: 0.8509
Epoch 30/300
435/435 - 343s - loss: 0.1428 - accuracy: 0.9520 - val_loss: 0.5539 - val_accuracy: 0.8323
Epoch 31/300
435/435 - 342s - loss: 0.1229 - accuracy: 0.9585 - val_loss: 0.5488 - val_accuracy: 0.8571
Epoch 32/300
435/435 - 342s - loss: 0.0955 - accuracy: 0.9673 - val_loss: 0.5816 - val_accuracy: 0.8634
Epoch 33/300
435/435 - 343s - loss: 0.0930 - accuracy: 0.9703 - val_loss: 0.5178 - val_accuracy: 0.8872
Epoch 34/300
435/435 - 343s - loss: 0.1100 - accuracy: 0.9634 - val_loss: 0.6582 - val_accuracy: 0.8530
Epoch 35/300
435/435 - 343s - loss: 0.0728 - accuracy: 0.9794 - val_loss: 0.6453 - val_accuracy: 0.8623
Epoch 36/300
435/435 - 343s - loss: 0.0645 - accuracy: 0.9787 - val_loss: 0.6742 - val_accuracy: 0.8561
Epoch 37/300
435/435 - 343s - loss: 0.0686 - accuracy: 0.9783 - val_loss: 0.7666 - val_accuracy: 0.8478
Epoch 38/300
435/435 - 343s - loss: 0.0727 - accuracy: 0.9755 - val_loss: 0.6426 - val_accuracy: 0.8375
Epoch 39/300
435/435 - 343s - loss: 0.0684 - accuracy: 0.9779 - val_loss: 0.8165 - val_accuracy: 0.8282
Epoch 40/300
435/435 - 342s - loss: 0.0673 - accuracy: 0.9777 - val_loss: 0.7447 - val_accuracy: 0.8540
Epoch 41/300
435/435 - 342s - loss: 0.0836 - accuracy: 0.9733 - val_loss: 0.7332 - val_accuracy: 0.8395
```

9. Training & Validation accuracy

![ ](https://raw.githubusercontent.com/ahoucbvtw/CNN-Kaggle-Pratice/main/Bees_Wasps_Insect_Other/Picture/Training%20%26%20Validation%20Accuracy.png)

10. Use **final_validation dataframe**  for final check to make a **Confusion_Matrix**

!["預測" = "Prediction" ; "真實" = "Real"](https://raw.githubusercontent.com/ahoucbvtw/CNN-Kaggle-Pratice/main/Bees_Wasps_Insect_Other/Picture/Confusion_Matrix.jpg)

ps : "預測" = "Prediction" ; "真實" = "Real"

11. Search relate picture URL and test this model's accuracy

![](https://github.com/ahoucbvtw/CNN-Kaggle-Pratice/blob/main/Bees_Wasps_Insect_Other/Picture/URL_Test.jpg?raw=true)


[Top](https://github.com/ahoucbvtw/CNN-Kaggle-Pratice#cnn-kaggle-pratice)
***
### HandGesture_Recognition Step Description
1. Unzip dataset

2. Make new training+testing folders and move pictures to these folders

3. Use **train_test_split** to separate final validation pictures. Each answer about 100 pictures.
4. make and move new folders for **final validation pictures**. And according to new path to make **final_df** 

![](https://raw.githubusercontent.com/ahoucbvtw/CNN-Kaggle-Pratice/main/HandGesture_Recognition/Picture/Final%20DataFrame.jpg)

5. Use Generate to preprocess images
```
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   zoom_range = 0.2,
                                   rotation_range = 30,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   fill_mode = "nearest")

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        path_Train,
        target_size = (250, 250),
        batch_size = 30,
        class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory(
        path_Test,
        target_size = (250, 250),
        batch_size = 50,
        class_mode = 'categorical')
```
6. Build CNN NetWork(self)
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_12 (Conv2D)           (None, 250, 250, 64)      1792      
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 250, 250, 128)     73856     
_________________________________________________________________
batch_normalization_7 (Batch (None, 250, 250, 128)     512       
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 125, 125, 128)     0         
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 125, 125, 128)     147584    
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 125, 125, 256)     295168    
_________________________________________________________________
batch_normalization_8 (Batch (None, 125, 125, 256)     1024      
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 62, 62, 256)       0         
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 62, 62, 256)       590080    
_________________________________________________________________
conv2d_17 (Conv2D)           (None, 62, 62, 512)       1180160   
_________________________________________________________________
batch_normalization_9 (Batch (None, 62, 62, 512)       2048      
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 31, 31, 512)       0         
_________________________________________________________________
conv2d_18 (Conv2D)           (None, 31, 31, 512)       2359808   
_________________________________________________________________
conv2d_19 (Conv2D)           (None, 31, 31, 1024)      4719616   
_________________________________________________________________
batch_normalization_10 (Batc (None, 31, 31, 1024)      4096      
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 15, 15, 1024)      0         
_________________________________________________________________
conv2d_20 (Conv2D)           (None, 15, 15, 1024)      9438208   
_________________________________________________________________
batch_normalization_11 (Batc (None, 15, 15, 1024)      4096      
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 7, 7, 1024)        0         
_________________________________________________________________
conv2d_21 (Conv2D)           (None, 7, 7, 500)         4608500   
_________________________________________________________________
batch_normalization_12 (Batc (None, 7, 7, 500)         2000      
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 3, 3, 500)         0         
_________________________________________________________________
conv2d_22 (Conv2D)           (None, 3, 3, 200)         900200    
_________________________________________________________________
conv2d_23 (Conv2D)           (None, 3, 3, 100)         180100    
_________________________________________________________________
batch_normalization_13 (Batc (None, 3, 3, 100)         400       
_________________________________________________________________
global_average_pooling2d_1 ( (None, 100)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
=================================================================
Total params: 24,514,808
Trainable params: 24,507,720
Non-trainable params: 7,088
_________________________________________________________________
```
7. Training
```
Epoch 1/150
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.1186s vs `on_train_batch_end` time: 0.2565s). Check your callbacks.
570/570 - 372s - loss: 1.3247 - accuracy: 0.5091 - val_loss: 5.3865 - val_accuracy: 0.1916
Epoch 2/150
570/570 - 359s - loss: 0.4325 - accuracy: 0.8557 - val_loss: 2.5981 - val_accuracy: 0.4526
Epoch 3/150
570/570 - 359s - loss: 0.2203 - accuracy: 0.9292 - val_loss: 0.6389 - val_accuracy: 0.7900
Epoch 4/150
570/570 - 360s - loss: 0.1375 - accuracy: 0.9582 - val_loss: 0.1359 - val_accuracy: 0.9511
Epoch 5/150
570/570 - 358s - loss: 0.1214 - accuracy: 0.9643 - val_loss: 0.6210 - val_accuracy: 0.8563
Epoch 6/150
570/570 - 359s - loss: 0.0943 - accuracy: 0.9714 - val_loss: 0.5880 - val_accuracy: 0.8479
Epoch 7/150
570/570 - 359s - loss: 0.0846 - accuracy: 0.9757 - val_loss: 1.1688 - val_accuracy: 0.6789
Epoch 8/150
570/570 - 361s - loss: 0.0802 - accuracy: 0.9757 - val_loss: 1.1788 - val_accuracy: 0.7695
Epoch 9/150
570/570 - 362s - loss: 0.0694 - accuracy: 0.9795 - val_loss: 0.0443 - val_accuracy: 0.9868
Epoch 10/150
570/570 - 360s - loss: 0.0578 - accuracy: 0.9833 - val_loss: 0.5593 - val_accuracy: 0.8705
Epoch 11/150
570/570 - 357s - loss: 0.0592 - accuracy: 0.9826 - val_loss: 1.8906 - val_accuracy: 0.7511
Epoch 12/150
570/570 - 359s - loss: 0.0442 - accuracy: 0.9877 - val_loss: 0.0729 - val_accuracy: 0.9726
Epoch 13/150
570/570 - 357s - loss: 0.0490 - accuracy: 0.9859 - val_loss: 0.3320 - val_accuracy: 0.9253
Epoch 14/150
570/570 - 359s - loss: 0.0435 - accuracy: 0.9871 - val_loss: 0.4850 - val_accuracy: 0.8942
Epoch 15/150
570/570 - 357s - loss: 0.0452 - accuracy: 0.9875 - val_loss: 3.4677 - val_accuracy: 0.4484
Epoch 16/150
570/570 - 362s - loss: 0.0368 - accuracy: 0.9891 - val_loss: 0.0083 - val_accuracy: 0.9989
Epoch 17/150
570/570 - 357s - loss: 0.0387 - accuracy: 0.9889 - val_loss: 0.9124 - val_accuracy: 0.8163
Epoch 18/150
570/570 - 360s - loss: 0.0379 - accuracy: 0.9898 - val_loss: 0.0117 - val_accuracy: 0.9979
Epoch 19/150
570/570 - 358s - loss: 0.0323 - accuracy: 0.9901 - val_loss: 3.1121 - val_accuracy: 0.5716
Epoch 20/150
570/570 - 359s - loss: 0.0292 - accuracy: 0.9917 - val_loss: 0.4636 - val_accuracy: 0.8958
Epoch 21/150
570/570 - 359s - loss: 0.0292 - accuracy: 0.9916 - val_loss: 0.2867 - val_accuracy: 0.9268
Epoch 22/150
570/570 - 362s - loss: 0.0256 - accuracy: 0.9934 - val_loss: 0.0025 - val_accuracy: 0.9995
Epoch 23/150
570/570 - 359s - loss: 0.0228 - accuracy: 0.9930 - val_loss: 0.0491 - val_accuracy: 0.9853
Epoch 24/150
570/570 - 359s - loss: 0.0237 - accuracy: 0.9936 - val_loss: 0.3088 - val_accuracy: 0.9195
Epoch 25/150
570/570 - 359s - loss: 0.0288 - accuracy: 0.9918 - val_loss: 0.0737 - val_accuracy: 0.9774
Epoch 26/150
570/570 - 358s - loss: 0.0247 - accuracy: 0.9937 - val_loss: 0.0034 - val_accuracy: 0.9989
Epoch 27/150
570/570 - 360s - loss: 0.0194 - accuracy: 0.9943 - val_loss: 0.0070 - val_accuracy: 0.9979
Epoch 28/150
570/570 - 358s - loss: 0.0258 - accuracy: 0.9929 - val_loss: 0.0031 - val_accuracy: 0.9984
Epoch 29/150

Epoch 00029: ReduceLROnPlateau reducing learning rate to 0.001.
570/570 - 358s - loss: 0.0177 - accuracy: 0.9946 - val_loss: 0.0033 - val_accuracy: 0.9989
Epoch 30/150
570/570 - 354s - loss: 0.0172 - accuracy: 0.9946 - val_loss: 0.0247 - val_accuracy: 0.9942
Epoch 31/150
570/570 - 358s - loss: 0.0228 - accuracy: 0.9936 - val_loss: 6.9813e-04 - val_accuracy: 1.0000
Epoch 32/150
570/570 - 357s - loss: 0.0190 - accuracy: 0.9950 - val_loss: 1.3806e-04 - val_accuracy: 1.0000
Epoch 33/150
570/570 - 355s - loss: 0.0123 - accuracy: 0.9961 - val_loss: 0.0404 - val_accuracy: 0.9858
Epoch 34/150
570/570 - 354s - loss: 0.0152 - accuracy: 0.9961 - val_loss: 0.0403 - val_accuracy: 0.9884
Epoch 35/150
570/570 - 356s - loss: 0.0217 - accuracy: 0.9942 - val_loss: 0.3463 - val_accuracy: 0.9421
Epoch 36/150
570/570 - 357s - loss: 0.0192 - accuracy: 0.9951 - val_loss: 0.7558 - val_accuracy: 0.8758
Epoch 37/150
570/570 - 359s - loss: 0.0189 - accuracy: 0.9943 - val_loss: 0.1161 - val_accuracy: 0.9737
Epoch 38/150

Epoch 00038: ReduceLROnPlateau reducing learning rate to 0.001.
570/570 - 358s - loss: 0.0192 - accuracy: 0.9947 - val_loss: 0.0065 - val_accuracy: 0.9979
Epoch 39/150
570/570 - 359s - loss: 0.0158 - accuracy: 0.9961 - val_loss: 0.0343 - val_accuracy: 0.9942
Epoch 40/150
570/570 - 357s - loss: 0.0123 - accuracy: 0.9967 - val_loss: 0.0228 - val_accuracy: 0.9947
Epoch 41/150
570/570 - 357s - loss: 0.0130 - accuracy: 0.9962 - val_loss: 0.0265 - val_accuracy: 0.9932
Epoch 42/150
570/570 - 357s - loss: 0.0175 - accuracy: 0.9951 - val_loss: 0.0016 - val_accuracy: 0.9995
```
8. Training & Validation accuracy

![ ](https://raw.githubusercontent.com/ahoucbvtw/CNN-Kaggle-Pratice/main/HandGesture_Recognition/Picture/Training%20%26%20Validation%20accuracy.png)

9. Training & Validation loss

![ ](https://raw.githubusercontent.com/ahoucbvtw/CNN-Kaggle-Pratice/main/HandGesture_Recognition/Picture/Training%20%26%20Validation%20loss.png)

10. Use **final_df** these new pictures to predict, to check model's accuracy, and make **Confusion_Matrix** 

![ ](https://raw.githubusercontent.com/ahoucbvtw/CNN-Kaggle-Pratice/main/HandGesture_Recognition/Picture/Confusion_Matrix.jpg)

ps : "預測" = "Prediction" ; "真實" = "Real"


[Top](https://github.com/ahoucbvtw/CNN-Kaggle-Pratice#cnn-kaggle-pratice)
***
### Intel Step Description
1. Unzip dataset

2. Because of pictures are all sorted, I can skip make folders and move pictures step, and use generate to preprocess images
```
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   zoom_range = 0.2,
                                   rotation_range = 30,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   fill_mode = "nearest")

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        path_Train,
        target_size = (150, 150),
        batch_size = 30,
        class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory(
        path_Test,
        target_size = (150, 150),
        batch_size = 50,
        class_mode = 'categorical')
```

3. Build CNN NetWork(self)
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 150, 150, 64)      1792      
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 150, 150, 128)     73856     
_________________________________________________________________
batch_normalization (BatchNo (None, 150, 150, 128)     512       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 75, 75, 128)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 75, 75, 128)       147584    
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 75, 75, 256)       295168    
_________________________________________________________________
batch_normalization_1 (Batch (None, 75, 75, 256)       1024      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 37, 37, 256)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 37, 37, 256)       590080    
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 37, 37, 512)       1180160   
_________________________________________________________________
batch_normalization_2 (Batch (None, 37, 37, 512)       2048      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 18, 18, 512)       0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 18, 18, 512)       2359808   
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 18, 18, 1024)      4719616   
_________________________________________________________________
batch_normalization_3 (Batch (None, 18, 18, 1024)      4096      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 9, 9, 1024)        0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 9, 9, 1024)        9438208   
_________________________________________________________________
batch_normalization_4 (Batch (None, 9, 9, 1024)        4096      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 4, 4, 1024)        0         
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 4, 4, 500)         4608500   
_________________________________________________________________
batch_normalization_5 (Batch (None, 4, 4, 500)         2000      
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 2, 2, 500)         0         
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 2, 2, 200)         900200    
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 2, 2, 100)         180100    
_________________________________________________________________
batch_normalization_6 (Batch (None, 2, 2, 100)         400       
_________________________________________________________________
global_average_pooling2d (Gl (None, 100)               0         
_________________________________________________________________
dropout (Dropout)            (None, 100)               0         
_________________________________________________________________
dense (Dense)                (None, 50)                5050      
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 306       
=================================================================
Total params: 24,514,604
Trainable params: 24,507,516
Non-trainable params: 7,088
_________________________________________________________________
```

4. Training
```
Epoch 1/150
468/468 - 155s - loss: 1.2804 - accuracy: 0.4781 - val_loss: 1.0291 - val_accuracy: 0.5563
Epoch 2/150
468/468 - 145s - loss: 1.0582 - accuracy: 0.5842 - val_loss: 1.3994 - val_accuracy: 0.5543
Epoch 3/150
468/468 - 148s - loss: 0.9331 - accuracy: 0.6489 - val_loss: 0.7039 - val_accuracy: 0.7420
Epoch 4/150
468/468 - 148s - loss: 0.8372 - accuracy: 0.6910 - val_loss: 1.0799 - val_accuracy: 0.6013
Epoch 5/150
468/468 - 147s - loss: 0.7693 - accuracy: 0.7211 - val_loss: 0.8618 - val_accuracy: 0.6807
Epoch 6/150
468/468 - 147s - loss: 0.7136 - accuracy: 0.7418 - val_loss: 0.9569 - val_accuracy: 0.6790
Epoch 7/150
468/468 - 147s - loss: 0.6868 - accuracy: 0.7520 - val_loss: 0.9223 - val_accuracy: 0.6707
Epoch 8/150
468/468 - 147s - loss: 0.6438 - accuracy: 0.7696 - val_loss: 0.7549 - val_accuracy: 0.7093
Epoch 9/150
468/468 - 147s - loss: 0.6330 - accuracy: 0.7752 - val_loss: 0.6453 - val_accuracy: 0.7773
Epoch 10/150
468/468 - 147s - loss: 0.5948 - accuracy: 0.7898 - val_loss: 0.5452 - val_accuracy: 0.7987
Epoch 11/150
468/468 - 148s - loss: 0.5825 - accuracy: 0.7919 - val_loss: 1.2776 - val_accuracy: 0.5897
Epoch 12/150
468/468 - 147s - loss: 0.5558 - accuracy: 0.8065 - val_loss: 0.5333 - val_accuracy: 0.8043
Epoch 13/150
468/468 - 147s - loss: 0.5428 - accuracy: 0.8101 - val_loss: 0.6207 - val_accuracy: 0.7783
Epoch 14/150
468/468 - 147s - loss: 0.5340 - accuracy: 0.8100 - val_loss: 0.4829 - val_accuracy: 0.8267
Epoch 15/150
468/468 - 147s - loss: 0.4998 - accuracy: 0.8241 - val_loss: 0.5028 - val_accuracy: 0.8273
Epoch 16/150
468/468 - 147s - loss: 0.4966 - accuracy: 0.8271 - val_loss: 0.8090 - val_accuracy: 0.6957
Epoch 17/150
468/468 - 147s - loss: 0.4844 - accuracy: 0.8300 - val_loss: 0.5222 - val_accuracy: 0.8173
Epoch 18/150
468/468 - 147s - loss: 0.4715 - accuracy: 0.8327 - val_loss: 0.4725 - val_accuracy: 0.8350
Epoch 19/150
468/468 - 147s - loss: 0.4523 - accuracy: 0.8393 - val_loss: 0.6628 - val_accuracy: 0.7733
Epoch 20/150
468/468 - 147s - loss: 0.4483 - accuracy: 0.8428 - val_loss: 0.7046 - val_accuracy: 0.7837
Epoch 21/150
468/468 - 147s - loss: 0.4285 - accuracy: 0.8495 - val_loss: 0.5825 - val_accuracy: 0.8003
Epoch 22/150
468/468 - 147s - loss: 0.4287 - accuracy: 0.8518 - val_loss: 0.9253 - val_accuracy: 0.7210
Epoch 23/150
468/468 - 147s - loss: 0.4161 - accuracy: 0.8546 - val_loss: 0.6175 - val_accuracy: 0.7873
Epoch 24/150
468/468 - 147s - loss: 0.4159 - accuracy: 0.8556 - val_loss: 0.6722 - val_accuracy: 0.7500
Epoch 25/150
468/468 - 147s - loss: 0.4125 - accuracy: 0.8573 - val_loss: 0.3815 - val_accuracy: 0.8663
Epoch 26/150
468/468 - 147s - loss: 0.4012 - accuracy: 0.8611 - val_loss: 0.4646 - val_accuracy: 0.8267
Epoch 27/150
468/468 - 147s - loss: 0.3844 - accuracy: 0.8668 - val_loss: 0.4152 - val_accuracy: 0.8537
Epoch 28/150
468/468 - 147s - loss: 0.3835 - accuracy: 0.8675 - val_loss: 0.5847 - val_accuracy: 0.7990
Epoch 29/150
468/468 - 147s - loss: 0.3852 - accuracy: 0.8671 - val_loss: 0.3727 - val_accuracy: 0.8727
Epoch 30/150
468/468 - 147s - loss: 0.3679 - accuracy: 0.8739 - val_loss: 0.5825 - val_accuracy: 0.8113
Epoch 31/150
468/468 - 147s - loss: 0.3674 - accuracy: 0.8708 - val_loss: 1.0878 - val_accuracy: 0.7000
Epoch 32/150
468/468 - 147s - loss: 0.3501 - accuracy: 0.8787 - val_loss: 0.5583 - val_accuracy: 0.8027
Epoch 33/150
468/468 - 147s - loss: 0.3502 - accuracy: 0.8785 - val_loss: 0.4249 - val_accuracy: 0.8557
Epoch 34/150
468/468 - 147s - loss: 0.3542 - accuracy: 0.8752 - val_loss: 0.3919 - val_accuracy: 0.8537
Epoch 35/150
468/468 - 147s - loss: 0.3314 - accuracy: 0.8851 - val_loss: 0.3421 - val_accuracy: 0.8827
Epoch 36/150
468/468 - 147s - loss: 0.3390 - accuracy: 0.8833 - val_loss: 0.3524 - val_accuracy: 0.8737
Epoch 37/150
468/468 - 147s - loss: 0.3340 - accuracy: 0.8849 - val_loss: 0.3747 - val_accuracy: 0.8603
Epoch 38/150
468/468 - 147s - loss: 0.3307 - accuracy: 0.8837 - val_loss: 0.4020 - val_accuracy: 0.8637
Epoch 39/150
468/468 - 147s - loss: 0.3165 - accuracy: 0.8920 - val_loss: 0.5394 - val_accuracy: 0.8197
Epoch 40/150
468/468 - 147s - loss: 0.3166 - accuracy: 0.8912 - val_loss: 0.3670 - val_accuracy: 0.8773
Epoch 41/150
468/468 - 147s - loss: 0.3110 - accuracy: 0.8919 - val_loss: 0.5412 - val_accuracy: 0.8197
Epoch 42/150
468/468 - 147s - loss: 0.3084 - accuracy: 0.8909 - val_loss: 0.3465 - val_accuracy: 0.8827

Epoch 00042: ReduceLROnPlateau reducing learning rate to 0.001.
Epoch 43/150
468/468 - 147s - loss: 0.3024 - accuracy: 0.8963 - val_loss: 0.3912 - val_accuracy: 0.8760
Epoch 44/150
468/468 - 147s - loss: 0.2928 - accuracy: 0.8975 - val_loss: 0.3480 - val_accuracy: 0.8787
Epoch 45/150
468/468 - 147s - loss: 0.2909 - accuracy: 0.8985 - val_loss: 0.3560 - val_accuracy: 0.8757
```

5. Training & Validation accuracy

![ ](https://raw.githubusercontent.com/ahoucbvtw/CNN-Kaggle-Pratice/main/Intel/Picture/Training%20%26%20Validation%20accuracy.png)

6. Training & Validation loss

![ ](https://raw.githubusercontent.com/ahoucbvtw/CNN-Kaggle-Pratice/main/Intel/Picture/Training%20%26%20Validation%20loss.png)

7. Use **pre_df** these new pictures to predict, to check model's accuracy, and make **Confusion_Matrix** 

![ ](https://raw.githubusercontent.com/ahoucbvtw/CNN-Kaggle-Pratice/main/Intel/Picture/Confusion_Matrix.jpg)

ps : "預測" = "Prediction" ; "真實" = "Real"

8. Search relate picture URL and test this model's accuracy

![ ](https://raw.githubusercontent.com/ahoucbvtw/CNN-Kaggle-Pratice/main/Intel/Picture/URL_Test.jpg)


[Top](https://github.com/ahoucbvtw/CNN-Kaggle-Pratice#cnn-kaggle-pratice)