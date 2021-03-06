% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/hello.R
\name{extractJuice}
\alias{extractJuice}
\title{extractJuice}
\usage{
extractJuice(X_train, Y_train, numFolds, parCV, numCores, seedNum, verbose_p,
  fn, fn_params)
}
\arguments{
\item{X_train}{Training Data (excludes the response/target we wish to predict
) that will be fed into the pipeline function.}

\item{Y_train}{Training Response/Target - The response/target that will be fed into the pipeline function.}

\item{numFolds}{Integer indicating the number of folds to use to extract predictions}

\item{parCV}{Boolean indicating whether to parallelize the extraction prodcedure.}

\item{numCores}{Integer indicating the number of cores to use when generating predictions.}

\item{seedNum}{Integer indicating the seed number. Using the same seed will generate the same folds.}

\item{verbose_p}{Boolean indicating if prediction extraction details should be printed out the screen.}

\item{fn}{The pipeline function. The pipeline function must take parameters training data, training response,
validation data, validation response. See examples for details.}

\item{fn_params}{Additional parameters to supply to the pipeline function. See examples for details.}
}
\value{
CV Fold Predictions
}
\description{
Returns cross validation predictions from folds (i.e. if 2 folds, then aggregate the predictions on
fold 2 after training only on fold 1 & the predictions on fold 1 after training only on fold 2). This
is helps tremendously in ensembling schemes such as stacked generalization.
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

xgbPipeline_regression_extraction <- function(X_train, Y_train, X_test, Y_test, params)
{
  Y_train <- as.numeric(Y_train) - 1
  xgbFit <- xgboost(data = as.matrix(X_train), label = Y_train, max.depth = params[1],
                    eta = params[2], nround = params[3], objective = "reg:linear", verbose = 1)
  predictions <- predict(xgbFit, newdata = as.matrix(X_test))
  return(predictions)
}

xgbPipeline_classification_extraction <- function(X_train, Y_train, X_test, Y_test, params)
{
  Y_train <- as.numeric(Y_train) - 1
  xgbFit <- xgboost(data = as.matrix(X_train), label = Y_train, max.depth = params[1],
                    eta = params[2], nround = params[3], objective = "binary:logistic", verbose = 1)
  predictions <- predict(xgbFit, newdata = as.matrix(X_test))
  return(predictions)
}

# Testing extractJuice
extractJuice(X_train = mtcarsMat, Y_train = mtcarsResponse, numFolds = 2,
             parCV = FALSE, numCores = 8, seedNum = 101, verbose_p = TRUE,
             fn = xgbPipeline_regression_extraction, fn_params = c(3, 2, 10))

extractJuice(X_train = irisTrainMat, Y_train = irisTrainResponse, numFolds = 2,
             parCV = FALSE, numCores = 8, seedNum = 101, verbose_p = TRUE,
             fn = xgbPipeline_classification_extraction, fn_params = c(3, 2, 10))
}

