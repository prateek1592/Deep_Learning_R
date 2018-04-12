## Self Neural Nets
## version 4 - L layer N with regularization

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

initialize_parameters <- function(X, Y, n_h=c(3,3)){
  out <- list()
  n_h <- c(nrow(X),n_h,nrow(Y))
  for (i in 2:length(n_h)){
    out[[paste("W",i-1,sep="")]] <- matrix(runif(n_h[i] * n_h[i-1]) * 0.01, nrow=n_h[i], ncol=n_h[i-1])
    out[[paste("B",i-1,sep="")]] <- matrix(0,nrow=n_h[i], ncol=1)
  }
  return(out)
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
  out <- list()
  num_layers <- length(parameters) / 2
  activation_func <- c(rep(activation_func,num_layers-1),"sigmoid")
  A <- X
  for (i in 1:num_layers){
    A_prev <- A
    out[[paste("Z",i,sep="")]] <- Z <- sweep(parameters[[paste("W",i,sep="")]] %*% A_prev, 1,
                                             parameters[[paste("B",i,sep="")]], '+')
    out[[paste("A",i,sep="")]] <- A <- activation_function(Z, type=activation_func[i])[["A"]]
  }
  return (out)
}

compute_cost <- function(Y, parameters, cache, lambda=0){
  num_layers <- length(cache) / 2
  prediction <- cache[[paste("A",num_layers,sep="")]]
  cost <- -1 / ncol(Y) * sum(Y * log(prediction) + (1-Y) * log(1-prediction))
  coefs <- parameters[grepl("W",names(parameters))]
  cost <- cost + lambda / (2 * ncol(Y)) * sum(unlist(lapply(coefs, function(x) sum(x^2))))
  cost
}

back_prop <- function(X, Y, parameters, cache, activation_func="tanh", lambda=0){
  out <- list()
  num_layers <- length(parameters) / 2
  activation_func <- c(rep(activation_func, num_layers-1),"sigmoid")
  cache[["A0"]] <- X
  dA <- -1 * (Y / cache[[paste("A",num_layers,sep="")]] - 
                (1-Y) / (1-cache[[paste("A",num_layers,sep="")]]))
  for (i in seq(num_layers,1,-1)){
    dZ <- dA * activation_function(cache[[paste("Z",i,sep="")]], type=activation_func[i])[["A_prime"]]
    out[[paste("dW",i,sep="")]] <- 1 / ncol(Y) * (dZ %*% t(cache[[paste("A",i-1,sep="")]])) +
      lambda / ncol(Y) * (parameters[[paste("W",i,sep="")]])
    out[[paste("dB",i,sep="")]] <- 1 / ncol(Y) * rowSums(dZ)
    dA <- t(parameters[[paste("W",i,sep="")]]) %*% dZ
  }
  return(out)
}

update_parameters <- function(parameters, gradients, learning_rate=0.01){
  num_layers <- length(parameters) / 2
  for (i in 1:num_layers){
    parameters[[paste("W",i,sep="")]] <- parameters[[paste("W",i,sep="")]] - 
      learning_rate * gradients[[paste("dW",i,sep="")]]
    parameters[[paste("B",i,sep="")]] <- parameters[[paste("B",i,sep="")]] - 
      learning_rate * gradients[[paste("dB",i,sep="")]]
  }
  return(parameters)
}

predict <- function(X, Y, parameters, activation_func="tanh"){
  pred <- forward_prop(X, parameters, activation_func=activation_func)
  pred <- pred[length(pred)][[1]]
  pred <- 1 * (pred > 0.5)
  accuracy <- sum(pred==Y) / length(pred)
  return (setNames(list(pred, accuracy),c("Prediction", "Test_Accuracy")))
}

cost_accuracy_plot <- function(J, test_accuracy){
  plot(J, type='l')
  par(new='T')
  plot(test_accuracy, col='red', type='o', yaxt='n', xaxt='n', ylab='', xlab='')
  axis(4)
  mtext('test_accuracy',side=4)
}

# # # # Execution # # # #

num_iter <- 5000
test_every <- 200
hidden_layer_activation <- "tanh"
lambda <- 0
n_h <- c(3)
alpha <- 0.01

data <- iris[1:100,]
data$Species <- as.numeric(data$Species) - 1
data <- preprocess_data(data, train_ratio = 0.7)
parameters <- initialize_parameters(data$Train$X, data$Train$Y, n_h = n_h)
test_accuracy <- J <- rep(NA, num_iter)

for (i in 1:num_iter){
  cache <- forward_prop(data$Train$X, parameters, activation_func = hidden_layer_activation)
  J[i] <- compute_cost(data$Train$Y, parameters, cache, lambda=lambda)
  gradients <- back_prop(data$Train$X, data$Train$Y, parameters, cache, 
                         lambda=lambda, activation_func = hidden_layer_activation)
  parameters <- update_parameters(parameters, gradients, learning_rate = alpha)
  if (!i %% test_every){
    list[prediction, accuracy] <- predict(data$Test$X, data$Test$Y, parameters, activation_func = hidden_layer_activation)
    test_accuracy[i] <- accuracy
  }
}

cost_accuracy_plot(J, test_accuracy)