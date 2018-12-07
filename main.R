
# Libraries ---------------------------------------------------------------

library(MASS)

install.packages("nnet")
library(nnet)

# Read data ---------------------------------------------------------------

mnist <- read.csv(file="mnist.csv")
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

# Ink ---------------------------------------------------------------------

# Get ink amount per datapoint
ink <- data.frame("label"=y, "ink"=apply(mnist[,2:785], 1, function(z) sum(z)))
mean <- tapply(ink[,2], ink[,1], mean)
sd <- tapply(ink[,2], ink[,1], sd)
mean  # prints the mean values
sd    # prints the standard deviations

# Using this feature to classify the pixel data will not be able to differentiate between some of the digits. Mainly because the mean values of most of 
# the classes are very close to each other. Besides the rather close means the standard deviations are also very high for most digits, therefore their 
# normal distributions are overlapping each other for large parts. Subsequently this feature will not be able to distinguish between these digits.

# Plot normal distributions
colors <- c("red", "blue", "darkgreen", "gold", "black", "yellow", "green", "purple", "pink", "orange")
labels <- c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
x <- seq(5000, 60000, length=100000)
plot(x, dnorm(x, mean=mean[1], sd=sd[1]), type="l", col=colors[1], xlab="Ink", ylab="Density", ylim=c(0,0.0001))
for(i in 2:10) {
  lines(x, dnorm(x,mean=mean[i], sd=sd[i]), col=colors[i])
}  
legend("topright", inset=.05, title="Digits",legend=labels, col=colors,lty=c(1,1,1,1,1,1,1,1,1,1))

# Using these normal distributions the classes can roughly be divided into 4 groups. These are digits:
# - 1
# - 4, 7 and 9
# - 2,3,5,6 and 8
# - 0

# Train multinomial model
ink.scaled <- data.frame("label"=ink[,1], "ink"=scale(ink[,2]))
mm <- multinom(label ~ ink, data = ink.scaled)

# TODO: predict digits using the multinom model...


# Ink per row ---------------------------------------------------------------------
# Idea for new feature: take the average for every row (or column) of pixels.


