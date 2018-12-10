
# Libraries ---------------------------------------------------------------

library(MASS)
library(class)

install.packages("nnet")
library(nnet)

install.packages("glmnet")
library(glmnet)
install.packages("tidyverse")
library(tidyverse)
install.packages("caret")
library(caret)

# Read data ---------------------------------------------------------------

mnist <- as.data.frame(read.csv(file="mnist.csv"))
mnist[,1] <- as.factor(mnist[,1])
y <- mnist[,1]
x <- mnist[,2:785]
summary(mnist[,1])

# findings:
# 
# The corner pixels often are 0 (mean is very low), therefore they dont really influence the classification process (superfluous).
# 
# > summary(mnist[,1])
# 0    1    2    3    4    5    6    7    8    9 
# 4132 4684 4177 4351 4072 3795 4137 4401 4063 4188
# 
# the dataset contains 42000 samples in total. Pictures with label "1" occur the most and those with label "5" 
# the least. there is a substantial difference between these labels 4684 - 3795 = 889....

normalDistPlot <- function(features, seqA, seqB, lim) {
  mean <- tapply(features[,2], features[,1], mean)
  sd <- tapply(features[,2], features[,1], sd)
  
  colors <- c("red", "blue", "darkgreen", "gold", "black", "yellow", "green", "purple", "pink", "orange")
  labels <- c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
  s <- seq(seqA, seqB, length=100000)
  plot(s, dnorm(s, mean=mean[1], sd=sd[1]), type="l", col=colors[1], xlab="Ink", ylab="Density", ylim=c(0,lim))
  for(i in 2:10) {
    lines(s, dnorm(s,mean=mean[i], sd=sd[i]), col=colors[i])
  }  
  legend("topright", inset=.05, title="Digits",legend=labels, col=colors,lty=c(1,1,1,1,1,1,1,1,1,1))
}

# Ink feature ---------------------------------------------------------------------

# Get ink amount per datapoint
ink <- apply(mnist[,2:785], 1, function(z) sum(z))
features <- data.frame("label"=y, "ink"=ink)
mean <- tapply(features[,2], features[,1], mean)
sd <- tapply(features[,2], features[,1], sd)
mean  # prints the mean values
sd    # prints the standard deviations

# Using this feature to classify the pixel data will not be able to differentiate between some of the digits. Mainly because the mean values of most of 
# the classes are very close to each other. Besides the rather close means the standard deviations are also very high for most digits, therefore their 
# normal distributions are overlapping each other for large parts. Subsequently this feature will not be able to distinguish between these digits.

# Plot normal distributions
normalDistPlot(features, 0, 60000, 0.0001)

# Using these normal distributions the classes can roughly be divided into 4 groups. These are digits:
# - 1
# - 4, 7 and 9
# - 2,3,5,6 and 8
# - 0

# Train multinomial model
features <- data.frame("label"=y, "ink"=scale(ink))
# Why do we scale the data? --> this centers the data around 0, which makes the training process easier. The derivative of the cost function
# will always be in the same proportions. It will normalize the effect of the different features. For example if one feature, on average, 
# has high values and the other low values, it could be possible that the higher values have more impact on the classification process. 
# Even though they should contribute equally.

test(features) # You have to load the test function first (see "All features" region)


# Ink per row ---------------------------------------------------------------------
# New feature: sum the ink per row of the greyscale images.
# This feature will be able to discriminate between digits that differ in their horizontal shape. For example ones will have very different 
# patterns compared to eights, because on average 1 has a narrow shape and 8 is more broad.
rowInk <- function(x) {
  m <- matrix(x, nrow=28)
  result <- c()
  for(i in 1:28){
    result <- append(result, as.numeric(sum(m[i,])))
  }
  return(result)
}

# This feature counts the number of rows that have ink
rowsWithInk <- function(x) {
  m <- matrix(x, nrow=28)
  result <- 0
  for(i in 1:28){
    if(sum(m[i,]) > 0) {
      result <- result + 1
    }
  }
  return(result)
}

width <- t(apply(mnist[,2:785], 1, function(z) rowInk(z)))
features <- data.frame("label"=y, "width"=width)
normalDistPlot(features, 0, 4000, 0.002)

numRows <- apply(mnist[,2:785], 1, function(z) rowsWithInk(z))
features <- data.frame("label"=y, "width"=numRows)
normalDistPlot(features, 0, 28, 0.2)

# Ink per col ---------------------------------------------------------------------
# New feature: This is the same as the ink per row feature, only this one is for columns.
colInk <- function(x) {
  m <- matrix(x, nrow=28)
  result <- c()
  for(i in 1:28){
    result <- append(result, as.numeric(sum(m[,i])))
  }
  return(result)
}

colsWithInk <- function(x) {
  m <- matrix(x, nrow=28)
  result <- 0
  for(i in 1:28){
    if(sum(m[,i]) > 0) {
      result <- result + 1
    }
  }
  return(result)
}

height <- t(apply(mnist[,2:785], 1, function(z) colInk(z)))
features <- data.frame("label"=y, "width"=height)
normalDistPlot(features, 0, 4000, 0.002)

numCols  <- apply(mnist[,2:785], 1, function(z) colsWithInk(z))
features <- data.frame("label"=y, "width"=numCols)
normalDistPlot(features, 0, 28, 1.2)

# Reduce size ---------------------------------------------------------------------
# Here we introduce a function that reduces the size of the images. from 28x28 --> 14x14 greyscale images.
# This is achieved by looping over every 2x2 block within the 28x28 image and take the average of that block.
halfSize <- function(x) {
  m <- matrix(x, nrow=28)
  result <- c()
  for(i in 0:195) {
    row <- floor(i/14)*2 + 1
    col <- (i %% 14) * 2 + 1
    temp <- c(m[row, col],
              m[row + 1, col],
              m[row, col + 1],
              m[row + 1, col + 1])
    result <- append(result, mean(temp))
  }
  return(result)
}

reducedImages <- t(apply(mnist[,2:785], 1, function(x) halfSize(x)))

# Remove superfluous ---------------------------------------------------------------------
removeSuperfluous <- function(data, npixels) {
  indices <- c()
  for(i in 1:npixels) {
    superfluous <- TRUE
    for(j in 1:nrow(data)) {
      if(data[j, i] > 0) superfluous <- FALSE
    }
    if(! superfluous) {
      indices <- append(indices, i)
    }
  }
  return(data[,indices])
}


reducedImages <- removeSuperfluous(reducedImages, 196)

# Classify ---------------------------------------------------------------------
# Here we can analyse all our features. First load the features that you want to test as a data.frame, then pass that to the test function.

test <- function(features, trainSize = 0.2, folds = 5, k = 5) {
  # Prepare data
  set.seed(123)
  training.samples <- features$label %>% createDataPartition(p = trainSize, list = FALSE)
  train.data <- features[training.samples,]
  test.data <- features[-training.samples,]
  train.x <- model.matrix(label~., train.data)[,-1]
  train.y <- train.data$label
  test.x <- model.matrix(label ~., test.data)[,-1]
  test.y <- test.data$label
  print("Prepared  the data")
  
  # Train multinomial logit model
  cv.fit <- getLambda(train.x, train.y, k = folds)
  multinom.model <- glmnet(train.x, train.y, alpha = 1, family="multinomial")
  # Test the model
  multinom.pred <- predict(multinom.model, test.x, s=cv.fit$lambda.min, type = "class")
  print("trained multinom model")
  
  # Predict using knn
  knn.pred <- knn(train.x, test.x, train.y, k = k)
  
  # print accuracies
  print("Accuracy per model:")
  print(paste("multinom:", mean(multinom.pred == test.y)))
  print(paste("knn:", mean(knn.pred == test.y)))
}

getLambda <- function(x, y, alpha = 1, k = 5) {
  return(cv.glmnet(x, y, alpha = alpha, family = "multinomial", nfolds=k, type.measure = "class", grouped = FALSE))
}


allFeatures <- data.frame("label"=y, "ink"=scale(ink), "width"=scale(width), "height"=scale(height), "numCols"=scale(numCols), "numRows"= scale(numRows))
features <- data.frame("label"=y, "ink"=scale(ink), "width"=scale(width))
features <- data.frame("label"=y, reducedImages)

features <- data.frame("label"=y, width, height)

test(features, trainSize = 0.1, k = 3)



# 1. What data pre-processing (including feature extraction/feature selection) did you perform for this classification algorithm? New 
#    features that you derive from the raw pixel data must be described unambiguously in the report. The reader should be able to 
#    reproduce your analysis. 
# First describe the extracted features (these are ink, averageRowInk, rowsWithInk, averageColInk, colsWithInk) TODO: add more features
# For every classification algorithm the data is split into train set and a test set. The trainset is a list of randomly selected samples 
# from the data. The size is equal to 10% of the total datapoints (this is 4200). 
# - Multinomial logit: a lambda value for the regularization has to be calculated. 
# - Knn: 
# - SVM: 

# 2. What are the complexity parameters (if any) of the classification algorithm, and how did you select their values?
# 3. What is the estimated accuracy of the best classifier for this method? 


