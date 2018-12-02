
# Libraries ---------------------------------------------------------------

library(MASS)

# Read data ---------------------------------------------------------------
train.dat <- read.csv(file="optdigits-tra.txt", sep=",")
test.dat <- read.csv(file="optdigits-tes.txt", sep=",")

summary(trainData)
summary(testData)
# The corner pixels often are 0 (mean is very low), therefore they dont really influence the classification process.

mnist <- read.csv(file="mnist.csv")
mnist[,1] <- as.factor(mnist[,1])
summary(mnist[,1])

# findings:
# 
# > summary(myData[,1])
# 0    1    2    3    4    5    6    7    8    9 
# 4132 4684 4177 4351 4072 3795 4137 4401 4063 4188
# 
# the dataset contains 42000 samples in total. Pictures with label "1" occur the most and those with label "5" 
# the least. there is a substantial difference between these labels 4684 - 3795 = 889....

# Analyse data ------------------------------------------------------------

# Get ink amount per datapoint
ink <- apply(mnist, 1, function(x) sum(x[-1]))
inkdf <- data.frame(label=mnist[,1],ink=ink)
