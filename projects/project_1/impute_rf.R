library(missForest)
library(doParallel)

setwd("/home/stefan/00_eth/aml/task1/ethz-advanced-ml/projects/project_1/")

X_train <- read.csv("X_train.csv")
X_test <- read.csv("X_test.csv")
X_train <- X_train[-1]
X_test <- X_test[-1]
X <- rbind(X_train, X_test)

numCores <- detectCores()
numCores
cl <- makeCluster(12)
registerDoParallel(cl)

X.imp <- missForest(X, parallelize = 'variables')
write.csv(X.imp$ximp, file = "X_imp100.csv")

X_train_imp <- X.imp$ximp[1:1212,]
X_test_imp <- X.imp$ximp[1213:1988,]

write.csv(X_train_imp, file = "X_train_imp100.csv")
write.csv(X_test_imp, file = "X_test_imp100.csv")
