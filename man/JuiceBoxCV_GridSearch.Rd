% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/hello.R
\name{JuiceBoxCV_GridSearch}
\alias{JuiceBoxCV_GridSearch}
\title{JuiceBoxCV_GridSearch}
\usage{
JuiceBoxCV_GridSearch(X_train, Y_train, numFolds, numRepeats, goParallel,
  numCores, seedNum, verbose_p, fitGrid, fn)
}
\arguments{
\item{X_train}{Training Data (excludes the response/target we wish to predict
) that will be fed into the pipeline function.}

\item{Y_train}{Training Response/Target - The response/target that will be fed into the pipeline function.}

\item{numFolds}{Integer indicating the number of folds to use in the cross validation procedure.}

\item{numRepeats}{Integer indicating the number of times to repeat cross validation with numFolds.}

\item{goParallel}{String indicating how to parallelize the procedure.
If "parGS", then the parallelization will happen across on the grid search across the parameters. If "parCV", then the parallelization will
happen across the numFolds and numRepeats cross validation procedure.}

\item{numCores}{Integer indicating the number of cores to use.}

\item{seedNum}{Integer indicating the seed number. Using the same seed will generate the same folds.}

\item{verbose_p}{Boolean indicating if grid search details should be printed out the screen.}

\item{fitGrid}{Dataframe dictating the combinations to try. See examples for details.}

\item{fn}{The pipeline function. The pipeline function must take parameters training data, training response,
validation data, validation response. See examples for details.}
}
\value{
Optimal parameters found from grid search
}
\description{
Use grid search in conjunction with cross validation to find optimal parameters for your pipline.
For each parameter, a cross validation procedure utilizing numFolds folds and numRepeats repetitions will
be executed. The parameter yielding largest CV score will be returned (i.e. maximization). In order to search for parameters that minimize the objective,
simply negate the value return in the pipeline function.
}
\examples{
library(JuiceBox)
library(Metrics)
library(xgboost)

# Toy data set for classification
irisAllMat <- iris
irisTrainMat <- irisAllMat[,c(1:4)]
irisTrainResponse <- irisAllMat[,c(ncol(irisAllMat))]
irisTrainResponse <- factor(ifelse(irisTrainResponse == "setosa", "Yes", "No"))

 # Toy data set for regression
mtcarsMat <- mtcars
mtcarsResponse <-mtcarsMat[,1]
mtcarsMat <- mtcarsMat[,c(2:ncol(mtcarsMat))]

# Pipelines
xgbPipeline_regression <- function(X_train, Y_train, X_test, Y_test, params)
{
  Y_train <- as.numeric(Y_train) - 1
  xgbFit <- xgboost(data = as.matrix(X_train), label = Y_train, max.depth = params[1],
                    eta = params[2], nround = params[3], objective = "reg:linear", verbose = 1)
  predictions <- predict(xgbFit, newdata = as.matrix(X_test))
  actual <- as.numeric(Y_test) - 1
  rmseValue <- rmse(predictions, actual)
  return(-rmseValue)
}

xgbPipeline_classification <- function(X_train, Y_train, X_test, Y_test, params)
{
  Y_train <- as.numeric(Y_train) - 1
  xgbFit <- xgboost(data = as.matrix(X_train), label = Y_train, max.depth = params[1],
                    eta = params[2], nround = params[3], objective = "binary:logistic", verbose = 1)
  predictions <- predict(xgbFit, newdata = as.matrix(X_test))
  actual <- as.numeric(Y_test) - 1
  logLossValue <- logLoss(actual, predictions)
  return(-logLossValue)
}

# Testing JuiceBoxCV_GridSearch
JuiceBoxCV_GridSearch(X_train = irisTrainMat, Y_train = irisTrainResponse, numFolds = 2,
                      numRepeats = 2, goParallel = "none", numCores = 8,
                      seedNum = 101, verbose_p = FALSE,
                      fitGrid = expand.grid(c(1:4), c(1:3), c(1:5)),
                      fn = xgbPipeline_classification)

JuiceBoxCV_GridSearch(X_train = mtcarsMat, Y_train = mtcarsResponse, numFolds = 2,
                      numRepeats = 2, goParallel = "none", numCores = 8,
                      seedNum = 101, verbose_p = FALSE,
                      fitGrid = expand.grid(c(1:4), c(1:3), c(1:5)),
                      fn = xgbPipeline_regression)
}

