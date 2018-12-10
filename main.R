
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

# Ink ---------------------------------------------------------------------

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
# New feature: average the ink per row of the greyscale images, only take rows into account that contain ink.
# This feature will be able to discriminate between digits that differ in their horizontal shape. For example ones will have very different 
# patterns compared to eights, because on average 1 has a narrow shape and 8 is more broad.
averageRowInk <- function(x) {
  m <- matrix(x, nrow=28)
  result <- c()
  for(i in 1:28){
    if(sum(m[i,]) > 0) {
      result <- append(result, as.numeric(sum(m[i,])))
    }
  }
  return(mean(result))
}

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

width <- apply(mnist[,2:785], 1, function(z) averageRowInk(z))
features <- data.frame("label"=y, "width"=width)
normalDistPlot(features, 0, 4000, 0.002)

numRows <- apply(mnist[,2:785], 1, function(z) rowsWithInk(z))
features <- data.frame("label"=y, "width"=numRows)
normalDistPlot(features, 0, 28, 0.2)

# Ink per col ---------------------------------------------------------------------
# New feature: This is the same as the ink per row feature, only this one is for columns.
averageColInk <- function(x) {
  m <- matrix(x, nrow=28)
  result <- c()
  for(i in 1:28){
    if(sum(m[,i]) > 0) {
      result <- append(result, as.numeric(sum(m[,i])))
    }
  }
  return(mean(result))
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

height <- apply(mnist[,2:785], 1, function(z) averageColInk(z))
features <- data.frame("label"=y, "width"=height)
normalDistPlot(features, 0, 4000, 0.002)

numCols  <- apply(mnist[,2:785], 1, function(z) colsWithInk(z))
features <- data.frame("label"=y, "width"=numCols)
normalDistPlot(features, 0, 28, 1.2)

# All features ---------------------------------------------------------------------
# Here we can analyse all our features. First load the features that you want to test as a data.frame, then pass that to the test function.
# getFoldIndices <- function(y, k = 10) {
#   require(caret)
#   return(createFolds(y, k = k, list = TRUE, returnTrain = FALSE))
# }

getLambda <- function(x, y, alpha = 1, k = 5) {
  return(cv.glmnet(x, y, alpha = alpha, family = "multinomial", nfolds=k, type.measure = "class", grouped = FALSE))
}

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
  
  # Train multinomial logit model
  cv.fit <- getLambda(train.x, train.y, k = folds)
  multinom.model <- glmnet(train.x, train.y, alpha = 1, family="multinomial")
  # Test the model
  multinom.pred <- predict(multinom.model, test.x, s=cv.fit$lambda.min, type = "class")
  
  
  # Predict using knn
  knn.pred <- knn(train.x, test.x, train.y, k = k)
  
  # print accuracies
  print("Accuracy per model:")
  print(paste("multinom:", mean(multinom.pred == test.y)))
  print(paste("knn:", mean(knn.pred == test.y)))
}


allFeatures <- data.frame("label"=y, "ink"=scale(ink), "width"=scale(width), "height"=scale(height), "numCols"=scale(numCols), "numRows"= scale(numRows))


features <- data.frame("label"=y, "ink"=scale(ink), "width"=scale(width))
test(features, k = 3)


