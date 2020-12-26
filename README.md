## CNN-Kaggle-Pratice
*** 
### List

[Bees_Wasps_Insect_Other Step Description](https://github.com/ahoucbvtw/CNN-Kaggle-Pratice#bees_wasps_insect_other-step-description)     

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
***