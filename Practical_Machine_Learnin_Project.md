Practical Machine Learning Course Project
================
W. Chen

Background
----------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

Data

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har>. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

The goal of the project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

``` r
knitr::opts_chunk$set(echo = TRUE)
```

Data Processing
---------------

Load the data from the web to R

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 3.1.3

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 3.1.3

``` r
library(rpart)
```

    ## Warning: package 'rpart' was built under R version 3.1.3

``` r
library(RColorBrewer)
```

    ## Warning: package 'RColorBrewer' was built under R version 3.1.3

``` r
library(randomForest)
```

    ## Warning: package 'randomForest' was built under R version 3.1.3

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(knitr)
TrainingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
OrigTrainingData <- read.csv(url(TrainingUrl), header=TRUE, sep=",", na.strings=c("NA","#DIV/0!",""))
TestingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
OrigTestingData <-  read.csv(url(TestingUrl), header=TRUE, sep=",", na.strings=c("NA","#DIV/0!",""))
```

Prepare the data
----------------

Prepare the data for the model

``` r
#Remove first 6 columns
TrainingData1 <- OrigTrainingData[, -(1:6)]
TestingData <- OrigTestingData[, -(1:6)]

#Remove NA columns
NA_columns<-colnames(TestingData)[colSums(is.na(TestingData)) > 0] 
TrainingData2<-TrainingData1[,!(names(TrainingData1) %in% NA_columns)]

#Remove NearZeroVariance columns
nzv <-nearZeroVar(TrainingData2, saveMetrics=TRUE)
TrainingData3 <- TrainingData2[,nzv$nzv==FALSE]
```

Divide the data 70/30 for the Cross Validation
----------------------------------------------

``` r
# Divide the training data into a training set (70%) and a validation set (30%)

Train <- createDataPartition(TrainingData3$classe, p = 0.7, list = FALSE)
MyTraining <- TrainingData3[Train,]
MyValidation <- TrainingData3[-Train,]
```

Create the training model by using Random Forest
------------------------------------------------

``` r
set.seed(2017)
# randomForest performs better
modFit <- randomForest(classe ~ ., data=MyTraining)
```

Accuracy on Training set
------------------------

``` r
PredTrining <- predict(modFit, MyTraining, type = "class")
confusionMatrix(PredTrining, MyTraining$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 3906    0    0    0    0
    ##          B    0 2658    0    0    0
    ##          C    0    0 2396    0    0
    ##          D    0    0    0 2252    0
    ##          E    0    0    0    0 2525
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9997, 1)
    ##     No Information Rate : 0.2843     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

``` r
plot(modFit)
```

![](Practical_Machine_Learnin_Project_files/figure-markdown_github/Accuracy%20on%20Training%20set-1.png) \#\# Accuracy on Validation set

``` r
PredValidation <- predict(modFit, MyValidation, type = "class")
confusionMatrix(PredValidation, MyValidation$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    3    0    0    0
    ##          B    0 1135    1    0    0
    ##          C    0    1 1025    5    0
    ##          D    0    0    0  958    1
    ##          E    1    0    0    1 1081
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9978          
    ##                  95% CI : (0.9962, 0.9988)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9972          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9965   0.9990   0.9938   0.9991
    ## Specificity            0.9993   0.9998   0.9988   0.9998   0.9996
    ## Pos Pred Value         0.9982   0.9991   0.9942   0.9990   0.9982
    ## Neg Pred Value         0.9998   0.9992   0.9998   0.9988   0.9998
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2843   0.1929   0.1742   0.1628   0.1837
    ## Detection Prevalence   0.2848   0.1930   0.1752   0.1630   0.1840
    ## Balanced Accuracy      0.9993   0.9981   0.9989   0.9968   0.9993

Prediction on the testing data
------------------------------

``` r
 predict(modFit, TestingData, type = "class")
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

End
---
