## Self Neural Nets
## version 1 - logistic regression

alpha <- 0.05
tol <- 0.1
chg <- Inf
data <- iris[1:60,]
n <- round(0.70*nrow(data))
skew_penalty <- 1    # For skew_penalty more than 1, classifying 1 as 0 == 'ghor apraadh'

data$Species <- as.numeric(data$Species) - 1
train_samples <- sample(nrow(data),n)
train <- data[train_samples,]
test <- data[-train_samples,]
y <- train$Species
x <- as.matrix(train[,-ncol(train)])

J <- b <- db <- iter <- 0
dw <- rep(0,4)
w <- rep(0.5,4)
dz <- a <- z <- numeric(n)

while (chg > tol){
  iter <- iter + 1
  for (i in 1:n){
    z[i] <- sum(w * x[i,]) + b
    a[i] <- 1 / (1 + exp(-z[i]))
    J <- J + -1*(skew_penalty * y[i] * log(a[i]) + (1-y[i]) * log(1-a[i]))
    dz[i] <- a[i] - skew_penalty * y[i] + (skew_penalty - 1) * a[i] * y[i]
    dw <- dw + dz[i] * x[i,]
    db <- db + dz[i]
  }
  J <- J/n
  dw <- dw/n
  db <- db/n
  
  prev_w <- w
  w <- w - alpha * dw
  b <- b - alpha * db
  chg <- sum(abs(w-prev_w))
  
}

actual <- test$Species
x_test <- as.matrix(test[,-ncol(test)])
pred <- as.numeric(1 / (1 + exp(-(x_test %*% w))))
pred <- pmax(sign(pred - 0.5),0)
sum(pred != actual) / length(pred)