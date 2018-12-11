
# Libraries ---------------------------------------------------------------

install.packages("e1071")
install.packages("nnet")
install.packages("glmnet")
install.packages("tidyverse")
install.packages("caret")

library(MASS)
library(class)
library(e1071)
library(nnet)
library(glmnet)
library(tidyverse)
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
# This function removes the superfluous features from the given data
removeSuperfluous <- function(data, nfeatures) {
  indices <- c()
  for(i in 1:nfeatures) {
    superfluous <- TRUE
    for(j in 1:nrow(data)) {
      if(data[j, i] != data[1,i]) superfluous <- FALSE
      if(!superfluous) break
    }
    if(! superfluous) {
      indices <- append(indices, i)
    }
  }
  return(data[,indices])
}


# Classify ---------------------------------------------------------------------
# Here we can analyse all our features. First load the features that you want to test as a data.frame, then pass that to the test function.

test <- function(features, trainSize = 0.2, folds = 5, k = 1:10, cost = 3:9, kernel = "radial") {
  print(paste("train size:", trainSize * nrow(features), "samples"))
  print(paste("test size:", nrow(features) - (trainSize * nrow(features)), "samples"))
  print(paste("Cross-validation folds:", folds))
  print(paste("SVM kernel:", kernel))
  # Prepare data
  set.seed(123)
  training.samples <- features$label %>% createDataPartition(p = trainSize, list = FALSE)
  train.data <- features[training.samples,]
  test.data <- features[-training.samples,]
  train.x <- model.matrix(label~., train.data)
  train.y <- train.data$label
  test.x <- model.matrix(label ~., test.data)
  test.y <- test.data$label
  print("Prepared  the data")
  
  # Multinomial logit
  print("Multinom logit model")
  start.time <- Sys.time()
  cv.fit <- getLambda(train.x, train.y, k = folds)
  multinom.model <- glmnet(train.x, train.y, alpha = 1, family="multinomial")
  end.time <- Sys.time()
  print(paste("Trained in", end.time - start.time, "seconds"))
  start.time <- Sys.time()
  multinom.pred <- predict(multinom.model, test.x, s=cv.fit$lambda.min, type = "class")
  end.time <- Sys.time()
  print(paste("Predicted in", end.time - start.time, "seconds"))
  
  # Knn
  print("Knn")
  knn.tune <- tune.knn(train.x, train.y, k = k)
  print(paste("Best k:", knn.tune$best.parameters[1,1]))
  start.time <- Sys.time()
  knn.pred <- knn(train.x, test.x, train.y, k = knn.tune$best.parameters[1,1])
  end.time <- Sys.time()
  print(paste("Predicted in", end.time - start.time, "seconds"))
  
  # SVM
  print("SVM")
  svm.tune <- tune.svm(train.x[,-1], train.y, cost = cost, kernel = kernel)
  print(paste("Best cost:", svm.tune$best.parameters[1,1]))
  start.time <- Sys.time()
  svm.model <- svm(train.x[,-1], train.y, cost = svm.tune$best.parameters[1,1], kernel = kernel)
  end.time <- Sys.time()
  print(paste("Trained in", end.time - start.time, "seconds"))
  start.time <- Sys.time()
  svm.pred <- predict(svm.model, test.x[,-1])
  end.time <- Sys.time()
  print(paste("Predicted in", end.time - start.time, "seconds"))
  
  # Print accuracies
  print("Accuracy per model:")
  print(paste("Multinom:", mean(multinom.pred == test.y)))
  print(paste("Knn:", mean(knn.pred == test.y)))
  print(paste("SVM:", sum(diag(table(test.y,svm.pred)))/nrow(test.data)))
}

getLambda <- function(x, y, alpha = 1, k = 5) {
  return(cv.glmnet(x, y, alpha = alpha, family = "multinomial", nfolds=k, type.measure = "class", grouped = FALSE))
}


feature.pixels <- scale(removeSuperfluous(x, 784))
feature.reducedPixels <- scale(removeSuperfluous(reducedImages, 28))
feature.width <- scale(removeSuperfluous(width,28))
feature.height <- scale(removeSuperfluous(height, 28))
feature.numCols <- scale(numCols)
feature.numRows <- scale(numRows)
feature.ink <- scale(ink)

features <- data.frame("label"=y, feature.width, feature.height)
test(features, trainSize = 0.2)


