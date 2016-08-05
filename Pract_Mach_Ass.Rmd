# Predicting the Manner the Excercise Is Being Performed

Sue, 
August 4, 2016

## Introduction

The main goal of this project is to predict the manner in which 6 participants performed some exercise. This is the "classe" variable in the training set. By doing some cross validation the best model is chosen and uses to predict the 20 cases in the test dataset.

##Data and Some Exploratory Analysis

In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available [here](http://groupware.les.inf.puc-rio.br/har). You can find the training and test data for this project here:

* [Trainset](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

* [Testset](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

## Loading and Cleaning Data

There are 160 variables in the original dataset many of which contain large number of NAs. Some other variables are highly correlated so we have to clean the data before further analysis. 

###Omitting Variables with High Amount of Missing Values
```{r, results= 'hide'}
library(dplyr)
library(caret)
library(tree)
library(randomForest)
library(rpart)
library(rpart.plot)
library(corrplot)

data= read.csv("./Prac_Mach_Ass.csv", header =T, stringsAsFactors = F )
data_test= read.csv("./Prac_Mach_Ass_test.csv", header =T, stringsAsFactors = F )

#choosing usefull variables:
  
data2=select(data,user_name, raw_timestamp_part_1, num_window, roll_belt, pitch_belt, yaw_belt, total_accel_belt, gyros_belt_x, gyros_belt_y, gyros_belt_z, accel_belt_x, accel_belt_y, accel_belt_z, magnet_belt_x, magnet_belt_y, magnet_belt_z, roll_arm, pitch_arm, yaw_arm, total_accel_arm, gyros_arm_y, gyros_arm_x, gyros_arm_z, accel_arm_x, accel_arm_y, accel_arm_z, magnet_arm_x, magnet_arm_y, magnet_arm_z, roll_dumbbell, pitch_dumbbell, yaw_dumbbell, total_accel_dumbbell, gyros_dumbbell_y, gyros_dumbbell_x, gyros_dumbbell_z, accel_dumbbell_x, accel_dumbbell_y, accel_dumbbell_z, magnet_dumbbell_x, magnet_dumbbell_y, magnet_dumbbell_z,roll_forearm, pitch_forearm, yaw_forearm, total_accel_forearm, gyros_forearm_y, gyros_forearm_x, gyros_forearm_z, accel_forearm_x, accel_forearm_y, accel_forearm_z, magnet_forearm_x, magnet_forearm_y, magnet_forearm_z,classe)

#Just numeric variables for further analysis

data2_c=select(data2, roll_belt, pitch_belt, yaw_belt, total_accel_belt, gyros_belt_x, gyros_belt_y, gyros_belt_z, accel_belt_x, accel_belt_y, accel_belt_z, magnet_belt_x, magnet_belt_y, magnet_belt_z, roll_arm, pitch_arm, yaw_arm, total_accel_arm, gyros_arm_y, gyros_arm_x, gyros_arm_z, accel_arm_x, accel_arm_y, accel_arm_z, magnet_arm_x, magnet_arm_y, magnet_arm_z, roll_dumbbell, pitch_dumbbell, yaw_dumbbell, total_accel_dumbbell, gyros_dumbbell_y, gyros_dumbbell_x, gyros_dumbbell_z, accel_dumbbell_x, accel_dumbbell_y, accel_dumbbell_z, magnet_dumbbell_x, magnet_dumbbell_y, magnet_dumbbell_z,roll_forearm, pitch_forearm, yaw_forearm, total_accel_forearm, gyros_forearm_y, gyros_forearm_x, gyros_forearm_z, accel_forearm_x, accel_forearm_y, accel_forearm_z, magnet_forearm_x, magnet_forearm_y, magnet_forearm_z)


```

### Finding Highly Correlated Variables

As we know high correlations between variables can ruin the model and predictions. Here I search for high correlations (more than 90%) and omit variables accordingly. The figure can show the correlations between variables much better. At the end just 46 out of 160 variable remain. 


```{r}
#Finding highly correlated variables
cor_columns = findCorrelation(cor(data2_c), cutoff = .89, verbose = TRUE)

#Removing correlated columns
data2_no_c = select(data2_c, -cor_columns)

#Checking near zero variable
zero_var=nearZeroVar(data2_no_c)
#No nearzero variance variable
#Check if there is any NA
sum(is.na(data2_no_c))
#No NA
corrplot(cor(data2_c), order = "FPC", method = "color", type = "lower", tl.cex = 0.6)
```

### Partitioning the Training Set for Some Cross Validation

```{r}

#adding classe variable to the rest of them
data_final=mutate(data2_no_c, classe=data2$classe)

#Split the data to training and testing data
inTrain=createDataPartition(data_final$classe, p=0.7, list = F)
Train=data_final[inTrain, ]
Test=data_final[-inTrain, ]
```

## Analysis

Here I use different methods for analysis. I mostly will use caret package.

### Decition Tree from Tree Package 
```{r, results= 'hide'}
#Analysis
#################################################rpart function##################################
set.seed(12345)
treeT<- rpart(classe ~ .,data=Train, method="class")
print(treeT)
```

```{r}
#Cross Validation
tree.P=predict(treeT,  newdata=Test, type="class")
confMatTR <- confusionMatrix(tree.P,Test$classe)
confMatTR

```
Not a promising prediction! Just 70% of accuracy!

### Tree Method from Caret Package 
```{r,  results= 'hide'}
#################################################rpart as method(more_clear)######################
set.seed(12345)
tree.Train2<- train(classe ~ .,method = "rpart", data=Train)
print(tree.Train2$finalModel)
plot(tree.Train2$finalModel)
text(tree.Train2$finalModel, use.n = T)
```

```{r}
#Cross Validation
tree.predict2=predict(tree.Train2, Test)
confMatTR2 <- confusionMatrix(tree.predict2, Test$classe)
confMatTR2

```

Prediction gets worse using caret package (only 49% of accuracy)!


### Random Forest from Caret Package

```{r, results= 'hide'}
#################################################Random Forest####################################
set.seed(12345)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
rf.Train=train(classe~., method = "rf", data=Train, trControl=controlRF)
rf.Train$finalModel
``` 

```{r}
#Cross Validation
predictRF <- predict(rf.Train, newdata=Test)
confMatRF <- confusionMatrix(predictRF, Test$classe)
confMatRF
```
The prediction is highly accurate (more than 99%) . This model is a good candidate to be used in predicting the test data set.

### Generalized Boosted Model from Caret

The last model I try is Generalized Boosted Model from caret package.

```{r, results= 'hide'}
#################################################Generalized Boosted Model########################
set.seed(12345)
controlgbm <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
gbm.Train  <- train(classe ~ ., data=Train, method = "gbm", trControl = controlgbm, verbose = FALSE)
gbm.Train$finalModel
```

```{r}
#Cross Validation
predictgbm <- predict(gbm.Train, newdata=Test)
confMatGBM <- confusionMatrix(predictgbm, Test$classe)
confMatGBM
```

The prediction of this model is also highly accurate (more than 96%), however is not as high as Random Forest method. Therefore I will use Random Forest for the last part of the project which is predicting the manner of exercise in test dataset.

```{r}
predictRF_test <- predict(rf.Train, newdata=data_test)
predictRF_test 
```
