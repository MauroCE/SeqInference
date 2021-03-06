% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/R_utils.R
\name{stochastic_volatility}
\alias{stochastic_volatility}
\title{Synthetic Stochastic Volatility Data}
\usage{
stochastic_volatility(nobs, param)
}
\arguments{
\item{nobs}{The number of observations to generate.}

\item{param}{A vector of model parameters used to generate the data.}
}
\value{
A data frame containing (x,y) pairs of observations. The states are 
given by x and the observations are given by y.
}
\description{
Synthetic Stochastic Volatility Data
}
