## Self Neural Nets
## version 2 - 2 Layer NN

# # # # Functions # # # #

list <- structure(NA,class="result")
"[<-.result" <- function(x,...,value){
  # Reference : https://stat.ethz.ch/pipermail/r-help/2004-June/053343.html
  args <- as.list(match.call())
  args <- args[-c(1:2,length(args))]
  length(value) <- length(args)
  for(i in seq(along=args)) {
    a <- args[[i]]
    if(!missing(a)) eval.parent(substitute(a <- v,list(a=a,v=value[[i]])))
  }
  x
}

revert_list_structure <- function(ls){
  x <- lapply(ls, `[`, names(ls[[1]]))
  apply(do.call(rbind, x), 2, as.list) 
}

preprocess_data <- function(data, train_ratio = 0.7, cv_ratio = 0){
  # Make sure that the last column is the dependent variable
  data <- data.frame(t(data)) # Need to transpose; observations should be columns!
  num_train <- train_ratio * ncol(data)
  train_samples <- sample(ncol(data), num_train)
  train <- data[,train_samples]
  non_train <- data[,-train_samples]
  num_cv <- cv_ratio * ncol(data)
  if (num_cv){
    cv_samples <- sample(ncol(non_train), num_cv)
    cv <- non_train[,cv_samples]
    test <- non_train[,-cv_samples]
  } else {
    cv <- NULL
    test <- non_train
  }
  
  all_data <- setNames(list(train, cv, test),c("Train", "CV", "Test"))
  all_data <- all_data[!sapply(all_data, is.null)]
  y <- lapply(all_data, function(df) as.matrix(df[nrow(df),]))
  x <- lapply(all_data, function(df) as.matrix(df[-nrow(df),]))
  return (revert_list_structure(setNames(list(x, y), c("X", "Y"))))
}

initialize_parameters <- function(X, Y, n_h=3){
  # Hidden Layer
  W1 <- matrix(runif(n_h * nrow(X)) * 0.01, nrow=n_h, ncol=nrow(X))
  B1 <- matrix(0,nrow=n_h, ncol=1)
  # Output Layer
  W2 <- matrix(runif(nrow(Y) * n_h) * 0.01, nrow=nrow(Y), ncol=n_h)
  B2 <- matrix(0,nrow=nrow(Y), ncol=1)
  return(setNames(list(W1, B1, W2, B2), c("W1", "B1", "W2", "B2")))
}

activation_function <- function(Z, type=c("sigmoid", "tanh", "relu", "leaky_relu"),
                                leaky_factor=0.01){
  type <- match.arg(type)
  if (type=="sigmoid"){
    A <- 1 / (1 + exp(-Z))
    A_prime <- A * (1-A)
  } else if (type=="tanh"){
    A <- (exp(Z) - exp(-Z)) / (exp(Z) + exp(-Z))
    A_prime <- 1-A^2
  } else if (type=="relu"){
    A <- pmax(Z, 0)
    A_prime <- 1 * (Z>0)
  } else if (type=="leaky_relu"){
    A <- pmax(Z, leaky_factor * Z)
    A_prime <- 1 * (Z>0) + leaky_factor * (Z<=0)
  }
  return (setNames(list(A, A_prime),c("A","A_prime")))
}

forward_prop <- function(X, parameters, activation_func="tanh"){
  list[W1,B1,W2,B2] <- parameters
  Z1 <- sweep(W1 %*% X, 1, B1, '+')
  A1 <- activation_function(Z1, type=activation_func)[["A"]]
  Z2 <- sweep(W2 %*% A1, 1, B2, '+')
  A2 <- activation_function(Z2, type="sigmoid")[["A"]]
  return (setNames(list(Z1, A1, Z2, A2),c("Z1", "A1", "Z2", "A2")))
}

compute_cost <- function(Y, cache){
  A2 <- cache[["A2"]]
  -1 / ncol(Y) * sum(Y * log(A2) + (1-Y) * log(1-A2))
}

back_prop <- function(X, Y, parameters, cache, activation_func="tanh"){
  list[W1,B1,W2,B2] <- parameters
  list[Z1,A1,Z2,A2] <- cache
  dZ2 <- (A2 - Y) / (A2 * (1-A2)) * activation_function(Z2, type="sigmoid")[["A_prime"]]
  dW2 <- 1 / ncol(Y) * (dZ2 %*% t(A1))
  dB2 <- 1 / ncol(Y) * rowSums(dZ2)
  dZ1 <- (t(W2) %*% dZ2) * activation_function(Z1, type=activation_func)[["A_prime"]]
  dW1 <- 1 / ncol(Y) * (dZ1 %*% t(X))
  dB1 <- 1 / ncol(Y) * rowSums(dZ1)
  return (setNames(list(dW1, dB1, dW2, dB2),c("dW1", "dB1", "dW2", "dB2")))
}

update_parameters <- function(parameters, gradients, learning_rate=0.01){
  list[W1,B1,W2,B2] <- parameters
  list[dW1,dB1,dW2,dB2] <- gradients
  W1 <- W1 - learning_rate * dW1
  W2 <- W2 - learning_rate * dW2
  B1 <- B1 - learning_rate * dB1
  B2 <- B2 - learning_rate * dB2
  return (setNames(list(W1, B1, W2, B2), c("W1", "B1", "W2", "B2")))
}

predict <- function(X, Y, parameters){
  pred <- forward_prop(X, parameters)
  pred <- pred[length(pred)][[1]]
  pred <- 1 * (pred > 0.5)
  accuracy <- sum(pred==Y) / length(pred)
  return (setNames(list(pred, accuracy),c("Prediction", "Test_Accuracy")))
}

cost_accuracy_plot <- function(J, test_accuracy){
  plot(J, type='l')
  par(new='T')
  plot(test_accuracy, col='red', type='o', yaxt='n', xaxt='n')
  axis(4)
}

# # # # Execution # # # #

num_iter <- 1000
test_every <- 50
hidden_layer_activation <- "relu"
data <- iris[1:100,]
data$Species <- as.numeric(data$Species) - 1
data <- preprocess_data(data, train_ratio = 0.7)
parameters <- initialize_parameters(data$Train$X, data$Train$Y, n_h = 3)
test_accuracy <- J <- rep(NA, num_iter)

for (i in 1:num_iter){
  cache <- forward_prop(data$Train$X, parameters, activation_func = hidden_layer_activation)
  J[i] <- compute_cost(data$Train$Y, cache)
  gradients <- back_prop(data$Train$X, data$Train$Y, parameters, cache, activation_func = hidden_layer_activation)
  parameters <- update_parameters(parameters, gradients, learning_rate = 0.01)
  if (!i %% test_every){
    list[prediction, accuracy] <- predict(data$Test$X, data$Test$Y, parameters)
    test_accuracy[i] <- accuracy
  }
}

cost_accuracy_plot(J, test_accuracy)