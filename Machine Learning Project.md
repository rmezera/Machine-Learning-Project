# Machine Learning Project


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(RandomForest)
```

```
## Error: there is no package called 'RandomForest'
```
## Read In Data

Here, I read in the training dataset to use in modeling.  I am also going to check what the dataset looks like and see what the Classe variable distribution is.


```r
pml.training <- read.csv("~/Downloads/pml-training.csv")
dim(pml.training)
```

```
## [1] 19622   160
```

```r
names(pml.training)
```

```
##   [1] "X"                        "user_name"               
##   [3] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
##   [5] "cvtd_timestamp"           "new_window"              
##   [7] "num_window"               "roll_belt"               
##   [9] "pitch_belt"               "yaw_belt"                
##  [11] "total_accel_belt"         "kurtosis_roll_belt"      
##  [13] "kurtosis_picth_belt"      "kurtosis_yaw_belt"       
##  [15] "skewness_roll_belt"       "skewness_roll_belt.1"    
##  [17] "skewness_yaw_belt"        "max_roll_belt"           
##  [19] "max_picth_belt"           "max_yaw_belt"            
##  [21] "min_roll_belt"            "min_pitch_belt"          
##  [23] "min_yaw_belt"             "amplitude_roll_belt"     
##  [25] "amplitude_pitch_belt"     "amplitude_yaw_belt"      
##  [27] "var_total_accel_belt"     "avg_roll_belt"           
##  [29] "stddev_roll_belt"         "var_roll_belt"           
##  [31] "avg_pitch_belt"           "stddev_pitch_belt"       
##  [33] "var_pitch_belt"           "avg_yaw_belt"            
##  [35] "stddev_yaw_belt"          "var_yaw_belt"            
##  [37] "gyros_belt_x"             "gyros_belt_y"            
##  [39] "gyros_belt_z"             "accel_belt_x"            
##  [41] "accel_belt_y"             "accel_belt_z"            
##  [43] "magnet_belt_x"            "magnet_belt_y"           
##  [45] "magnet_belt_z"            "roll_arm"                
##  [47] "pitch_arm"                "yaw_arm"                 
##  [49] "total_accel_arm"          "var_accel_arm"           
##  [51] "avg_roll_arm"             "stddev_roll_arm"         
##  [53] "var_roll_arm"             "avg_pitch_arm"           
##  [55] "stddev_pitch_arm"         "var_pitch_arm"           
##  [57] "avg_yaw_arm"              "stddev_yaw_arm"          
##  [59] "var_yaw_arm"              "gyros_arm_x"             
##  [61] "gyros_arm_y"              "gyros_arm_z"             
##  [63] "accel_arm_x"              "accel_arm_y"             
##  [65] "accel_arm_z"              "magnet_arm_x"            
##  [67] "magnet_arm_y"             "magnet_arm_z"            
##  [69] "kurtosis_roll_arm"        "kurtosis_picth_arm"      
##  [71] "kurtosis_yaw_arm"         "skewness_roll_arm"       
##  [73] "skewness_pitch_arm"       "skewness_yaw_arm"        
##  [75] "max_roll_arm"             "max_picth_arm"           
##  [77] "max_yaw_arm"              "min_roll_arm"            
##  [79] "min_pitch_arm"            "min_yaw_arm"             
##  [81] "amplitude_roll_arm"       "amplitude_pitch_arm"     
##  [83] "amplitude_yaw_arm"        "roll_dumbbell"           
##  [85] "pitch_dumbbell"           "yaw_dumbbell"            
##  [87] "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
##  [89] "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"  
##  [91] "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"   
##  [93] "max_roll_dumbbell"        "max_picth_dumbbell"      
##  [95] "max_yaw_dumbbell"         "min_roll_dumbbell"       
##  [97] "min_pitch_dumbbell"       "min_yaw_dumbbell"        
##  [99] "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
## [101] "amplitude_yaw_dumbbell"   "total_accel_dumbbell"    
## [103] "var_accel_dumbbell"       "avg_roll_dumbbell"       
## [105] "stddev_roll_dumbbell"     "var_roll_dumbbell"       
## [107] "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
## [109] "var_pitch_dumbbell"       "avg_yaw_dumbbell"        
## [111] "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        
## [113] "gyros_dumbbell_x"         "gyros_dumbbell_y"        
## [115] "gyros_dumbbell_z"         "accel_dumbbell_x"        
## [117] "accel_dumbbell_y"         "accel_dumbbell_z"        
## [119] "magnet_dumbbell_x"        "magnet_dumbbell_y"       
## [121] "magnet_dumbbell_z"        "roll_forearm"            
## [123] "pitch_forearm"            "yaw_forearm"             
## [125] "kurtosis_roll_forearm"    "kurtosis_picth_forearm"  
## [127] "kurtosis_yaw_forearm"     "skewness_roll_forearm"   
## [129] "skewness_pitch_forearm"   "skewness_yaw_forearm"    
## [131] "max_roll_forearm"         "max_picth_forearm"       
## [133] "max_yaw_forearm"          "min_roll_forearm"        
## [135] "min_pitch_forearm"        "min_yaw_forearm"         
## [137] "amplitude_roll_forearm"   "amplitude_pitch_forearm" 
## [139] "amplitude_yaw_forearm"    "total_accel_forearm"     
## [141] "var_accel_forearm"        "avg_roll_forearm"        
## [143] "stddev_roll_forearm"      "var_roll_forearm"        
## [145] "avg_pitch_forearm"        "stddev_pitch_forearm"    
## [147] "var_pitch_forearm"        "avg_yaw_forearm"         
## [149] "stddev_yaw_forearm"       "var_yaw_forearm"         
## [151] "gyros_forearm_x"          "gyros_forearm_y"         
## [153] "gyros_forearm_z"          "accel_forearm_x"         
## [155] "accel_forearm_y"          "accel_forearm_z"         
## [157] "magnet_forearm_x"         "magnet_forearm_y"        
## [159] "magnet_forearm_z"         "classe"
```

```r
table(pml.training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```
## Clean data

Now, I am going to remove the items that have a lot of missing/NA data points and name it my.training This will allow the analysis to run and also make my data files more manageable.  I found the missing ones by looking at the dataset.


```r
my.training <- pml.training[,c(6:11,37:49,60:68,84:86,102,113:124,140,151:160)]
```
## Prepare training data by spliting it up into a Train and Test

Split up Training Set into a Training and Testing

```r
set.seed(1432)
inTrain = createDataPartition(my.training$classe, p=.70)[[1]]
training = my.training[inTrain,]
testing = my.training[-inTrain,]
dim(training); dim(testing)
```

```
## [1] 13737    55
```

```
## [1] 5885   55
```
## Choosing a model to use
I first tried CART analysis on this but it didn't do a very good job so I left the output off this .Rmd document.

I heard that RandomForest is good at predicting so I am going to try to use the rf option in train (from the caret package).  Professors mentioned that Random Forest is used in many competitions so I thought that would be a good fit for this since accuracy was key and explaining the exact model was not the main goal of this.


```r
#modFitRF <- train(classe~.,methods="rf",data#=training)
#print(modFitRF$finalModel)
```
Now that we have the RandomForest model (modFitRF), we should check to see how it did at predicting the training and testing sets


```r
#table(training$classse,predict(modFitRF,training))
#table(testing$classe, predict(modFitRF,testing))
#predRF.tr <- predict(modFitRF,training)
#accuracyRF.tr = sum(predRF.tr ==training$classe)/length(predRF.tr)
#accuracyRF.tr
#predRF.te <- predict(modFitRF,testing)
#accuracyRF.te = sum(predRF.te ==testing$classe)/length(predRF.te)
#accuracyRF.te
```
Due to time contraints, the above code was not included when knitting because it took too long and it was not the one I ended up using.  But, while initially doing the analysis I did do the code above along with a CART analysis.

## Using Cross Validation with Random Forest model
The assignment asked to use Cross Validation so this is below using Random Forest with Cross Validation.  I am using a K=5 to add some accuracy but to not have it take forever on my computer.  That means we use 80% of the data each time so that seems reasonable to me. Plus, I already split out 70% into my training set so I have a set to test the error on after coming up with my model.


```r
ctrl <- trainControl(method = "cv", number  = 5)
modFitRF2 = train(classe~ . , data = training, method = 'rf', trControl = ctrl)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
modFitRF2$results
```

```
##   mtry Accuracy  Kappa AccuracySD  KappaSD
## 1    2   0.9931 0.9913   0.001838 0.002325
## 2   28   0.9975 0.9968   0.001151 0.001457
## 3   54   0.9937 0.9920   0.002504 0.003168
```
## Checking accuracy on the Test holdout set

I ran the model with Cross Validation and now I want to check the accuracy on my test set 

```r
predRF2 = predict(modFitRF2, testing)
table(predRF2,testing$classe)
```

```
##        
## predRF2    A    B    C    D    E
##       A 1673    1    0    0    0
##       B    1 1138    1    0    0
##       C    0    0 1025    5    0
##       D    0    0    0  959    6
##       E    0    0    0    0 1076
```

```r
accuracyRF2 = sum(predRF2 ==testing$classe)/length(predRF2)
accuracyRF2
```

```
## [1] 0.9976
```
The accuracyRF2 above is my best guess at the out of sample error since it is based on a holdout sample I had in the original Training set. Since the model was not built using this data it should be a good representation of the out of sample error.

In my opinion, this Model does a great job of predicting the classe variable and I will use that for predicting the 20 Test sets.

## Predicting the 20 Test sets using the Random Forest model I developed above.

I first need to read in the Test data and I made it look like the test set I used by removing the variables that were NA and missing in the Training set.

```r
pml.testing <- read.csv("~/Downloads/pml-testing.csv")
pml.testing2 <- pml.testing[,c(6:11,37:49,60:68,84:86,102,113:124,140,151:160)]
pml.testing2[1:2,]
```

```
##   new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
## 1         no         74    123.00      27.00    -4.75               20
## 2         no        431      1.02       4.87   -88.90                4
##   gyros_belt_x gyros_belt_y gyros_belt_z accel_belt_x accel_belt_y
## 1        -0.50        -0.02        -0.46          -38           69
## 2        -0.06        -0.02        -0.07          -13           11
##   accel_belt_z magnet_belt_x magnet_belt_y magnet_belt_z roll_arm
## 1         -179           -13           581          -382     40.7
## 2           39            43           636          -309      0.0
##   pitch_arm yaw_arm total_accel_arm gyros_arm_x gyros_arm_y gyros_arm_z
## 1     -27.8     178              10       -1.65        0.48       -0.18
## 2       0.0       0              38       -1.17        0.85       -0.43
##   accel_arm_x accel_arm_y accel_arm_z magnet_arm_x magnet_arm_y
## 1          16          38          93         -326          385
## 2        -290         215         -90         -325          447
##   magnet_arm_z roll_dumbbell pitch_dumbbell yaw_dumbbell
## 1          481        -17.74          24.96       126.24
## 2          434         54.48         -53.70       -75.51
##   total_accel_dumbbell gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z
## 1                    9             0.64             0.06            -0.61
## 2                   31             0.34             0.05            -0.71
##   accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x
## 1               21              -15               81               523
## 2             -153              155             -205              -502
##   magnet_dumbbell_y magnet_dumbbell_z roll_forearm pitch_forearm
## 1              -528               -56          141          49.3
## 2               388               -36          109         -17.6
##   yaw_forearm total_accel_forearm gyros_forearm_x gyros_forearm_y
## 1         156                  33            0.74           -3.34
## 2         106                  39            1.12           -2.78
##   gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
## 1           -0.59            -110             267            -149
## 2           -0.18             212             297            -118
##   magnet_forearm_x magnet_forearm_y magnet_forearm_z problem_id
## 1             -714              419              617          1
## 2             -237              791              873          2
```

```r
dim(pml.testing2)
```

```
## [1] 20 55
```

```r
names(pml.testing)
```

```
##   [1] "X"                        "user_name"               
##   [3] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
##   [5] "cvtd_timestamp"           "new_window"              
##   [7] "num_window"               "roll_belt"               
##   [9] "pitch_belt"               "yaw_belt"                
##  [11] "total_accel_belt"         "kurtosis_roll_belt"      
##  [13] "kurtosis_picth_belt"      "kurtosis_yaw_belt"       
##  [15] "skewness_roll_belt"       "skewness_roll_belt.1"    
##  [17] "skewness_yaw_belt"        "max_roll_belt"           
##  [19] "max_picth_belt"           "max_yaw_belt"            
##  [21] "min_roll_belt"            "min_pitch_belt"          
##  [23] "min_yaw_belt"             "amplitude_roll_belt"     
##  [25] "amplitude_pitch_belt"     "amplitude_yaw_belt"      
##  [27] "var_total_accel_belt"     "avg_roll_belt"           
##  [29] "stddev_roll_belt"         "var_roll_belt"           
##  [31] "avg_pitch_belt"           "stddev_pitch_belt"       
##  [33] "var_pitch_belt"           "avg_yaw_belt"            
##  [35] "stddev_yaw_belt"          "var_yaw_belt"            
##  [37] "gyros_belt_x"             "gyros_belt_y"            
##  [39] "gyros_belt_z"             "accel_belt_x"            
##  [41] "accel_belt_y"             "accel_belt_z"            
##  [43] "magnet_belt_x"            "magnet_belt_y"           
##  [45] "magnet_belt_z"            "roll_arm"                
##  [47] "pitch_arm"                "yaw_arm"                 
##  [49] "total_accel_arm"          "var_accel_arm"           
##  [51] "avg_roll_arm"             "stddev_roll_arm"         
##  [53] "var_roll_arm"             "avg_pitch_arm"           
##  [55] "stddev_pitch_arm"         "var_pitch_arm"           
##  [57] "avg_yaw_arm"              "stddev_yaw_arm"          
##  [59] "var_yaw_arm"              "gyros_arm_x"             
##  [61] "gyros_arm_y"              "gyros_arm_z"             
##  [63] "accel_arm_x"              "accel_arm_y"             
##  [65] "accel_arm_z"              "magnet_arm_x"            
##  [67] "magnet_arm_y"             "magnet_arm_z"            
##  [69] "kurtosis_roll_arm"        "kurtosis_picth_arm"      
##  [71] "kurtosis_yaw_arm"         "skewness_roll_arm"       
##  [73] "skewness_pitch_arm"       "skewness_yaw_arm"        
##  [75] "max_roll_arm"             "max_picth_arm"           
##  [77] "max_yaw_arm"              "min_roll_arm"            
##  [79] "min_pitch_arm"            "min_yaw_arm"             
##  [81] "amplitude_roll_arm"       "amplitude_pitch_arm"     
##  [83] "amplitude_yaw_arm"        "roll_dumbbell"           
##  [85] "pitch_dumbbell"           "yaw_dumbbell"            
##  [87] "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
##  [89] "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"  
##  [91] "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"   
##  [93] "max_roll_dumbbell"        "max_picth_dumbbell"      
##  [95] "max_yaw_dumbbell"         "min_roll_dumbbell"       
##  [97] "min_pitch_dumbbell"       "min_yaw_dumbbell"        
##  [99] "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
## [101] "amplitude_yaw_dumbbell"   "total_accel_dumbbell"    
## [103] "var_accel_dumbbell"       "avg_roll_dumbbell"       
## [105] "stddev_roll_dumbbell"     "var_roll_dumbbell"       
## [107] "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
## [109] "var_pitch_dumbbell"       "avg_yaw_dumbbell"        
## [111] "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        
## [113] "gyros_dumbbell_x"         "gyros_dumbbell_y"        
## [115] "gyros_dumbbell_z"         "accel_dumbbell_x"        
## [117] "accel_dumbbell_y"         "accel_dumbbell_z"        
## [119] "magnet_dumbbell_x"        "magnet_dumbbell_y"       
## [121] "magnet_dumbbell_z"        "roll_forearm"            
## [123] "pitch_forearm"            "yaw_forearm"             
## [125] "kurtosis_roll_forearm"    "kurtosis_picth_forearm"  
## [127] "kurtosis_yaw_forearm"     "skewness_roll_forearm"   
## [129] "skewness_pitch_forearm"   "skewness_yaw_forearm"    
## [131] "max_roll_forearm"         "max_picth_forearm"       
## [133] "max_yaw_forearm"          "min_roll_forearm"        
## [135] "min_pitch_forearm"        "min_yaw_forearm"         
## [137] "amplitude_roll_forearm"   "amplitude_pitch_forearm" 
## [139] "amplitude_yaw_forearm"    "total_accel_forearm"     
## [141] "var_accel_forearm"        "avg_roll_forearm"        
## [143] "stddev_roll_forearm"      "var_roll_forearm"        
## [145] "avg_pitch_forearm"        "stddev_pitch_forearm"    
## [147] "var_pitch_forearm"        "avg_yaw_forearm"         
## [149] "stddev_yaw_forearm"       "var_yaw_forearm"         
## [151] "gyros_forearm_x"          "gyros_forearm_y"         
## [153] "gyros_forearm_z"          "accel_forearm_x"         
## [155] "accel_forearm_y"          "accel_forearm_z"         
## [157] "magnet_forearm_x"         "magnet_forearm_y"        
## [159] "magnet_forearm_z"         "problem_id"
```

```r
names(training)
```

```
##  [1] "new_window"           "num_window"           "roll_belt"           
##  [4] "pitch_belt"           "yaw_belt"             "total_accel_belt"    
##  [7] "gyros_belt_x"         "gyros_belt_y"         "gyros_belt_z"        
## [10] "accel_belt_x"         "accel_belt_y"         "accel_belt_z"        
## [13] "magnet_belt_x"        "magnet_belt_y"        "magnet_belt_z"       
## [16] "roll_arm"             "pitch_arm"            "yaw_arm"             
## [19] "total_accel_arm"      "gyros_arm_x"          "gyros_arm_y"         
## [22] "gyros_arm_z"          "accel_arm_x"          "accel_arm_y"         
## [25] "accel_arm_z"          "magnet_arm_x"         "magnet_arm_y"        
## [28] "magnet_arm_z"         "roll_dumbbell"        "pitch_dumbbell"      
## [31] "yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_x"    
## [34] "gyros_dumbbell_y"     "gyros_dumbbell_z"     "accel_dumbbell_x"    
## [37] "accel_dumbbell_y"     "accel_dumbbell_z"     "magnet_dumbbell_x"   
## [40] "magnet_dumbbell_y"    "magnet_dumbbell_z"    "roll_forearm"        
## [43] "pitch_forearm"        "yaw_forearm"          "total_accel_forearm" 
## [46] "gyros_forearm_x"      "gyros_forearm_y"      "gyros_forearm_z"     
## [49] "accel_forearm_x"      "accel_forearm_y"      "accel_forearm_z"     
## [52] "magnet_forearm_x"     "magnet_forearm_y"     "magnet_forearm_z"    
## [55] "classe"
```

Here are my predictions for the 20 records in the test set. These have been submitted to the prediction submissions.


```r
predRF2.test = predict(modFitRF2,pml.testing2)
predRF2.test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```






