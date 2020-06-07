// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"
using namespace Rcpp;

class BS_Model {
  private:
    // ATTRIBUTES:
    arma::rowvec _param; // Model parameters (alpha, beta, sigma).
    arma::vec _obs; // Vector of observations to be filtered.
    arma::mat _samples; // Each column represents a draw from full posterior.
    int _N; // Number of particles.
    
  public:
    // CONSTRUCTOR:
    BS_Model(arma::vec obs, int N, arma::rowvec param) {
      int nobs = obs.n_elem;
      this->_param = param;
      this->_N = N;
      this->_samples = arma::zeros(nobs + 1, N);
      this->_obs = obs;
      
      // Initialise x0:
      double x0_sd = sqrt(pow(param(2), 2) / (1 - pow(param(0), 2)));
      NumericVector x0 = rnorm(N, 0.0, x0_sd);
      this->_samples.row(0) = as<arma::rowvec>(x0);
    }
    
    // METHODS:
    void transition(int t) {
      arma::rowvec x_new(this->_N);
      arma::rowvec x_prev = this->_samples.row(t - 1);
      for (int i = 0; i < this->_N; ++i) {
        x_new(i) = this->_param(0) * x_prev(i) + rnorm(1, 0.0, this->_param(2))[0];
      }
      this->_samples.row(t) = x_new;
    }
    
    void resample(int t, arma::vec weights) {
      // Ensure weights are nonzero:
      arma::uvec index = which_nonzero(weights);
      arma::vec weights_new = weights.elem(index);
      weights_new = normalize(weights_new);
      arma::vec to_sample = arma::linspace(0, this->_N - 1, this->_N);
      arma::vec to_sample_new = to_sample.elem(index);
      NumericVector t_s = wrap(to_sample_new);
      NumericVector probs = wrap(weights_new);
      NumericVector resampled = Rcpp::sample(t_s, this->_N, true, probs); // INVESTIGATE IF PROBS IS THE SAME LENGTH.
      arma::uvec columns = as<arma::uvec>(resampled);
      this->_samples = this->_samples.cols(columns);
    }
    
    arma::colvec sample(arma::vec weights) {
      arma::vec to_sample = arma::linspace(0, this->_N - 1, this->_N);
      NumericVector t_s = wrap(to_sample);
      NumericVector probs = wrap(weights);
      int index = Rcpp::sample(t_s, 1, false, probs)[0];
      return this->_samples.col(index);
    }
    
    arma::vec likelihood(int t) {
      arma::rowvec x = this->_samples.row(t);
      arma::vec lik(this->_N);
      NumericVector y(1);
      y[0] = this->_obs(t - 1); // Account for x0 by subtracting 1.
      double lik_sd;
      for (int i = 0; i < this->_N; ++i) {
        lik_sd = this->_param(1) * exp(x(i) / 2);
        lik(i) = dnorm(y, 0.0, lik_sd)[0];
      }
      return lik;
    }
    
    // attribute access...
    arma::mat getSamples() {
      return this->_samples;
    }
  
};

//' Bootstrap Particle Filter
//' 
//' @param obs A vector of observations to be filtered.
//' @param N The number of particles to be used in the filter.
//' @param param The model parameters to be used within the filter. These are
//' passed in the form c(alpha, beta, sigma).
//' 
//' @return A list containing the filtered states; a sample from the posterior
//' distribution; the log-marginal distribution of the observations.
//' 
// [[Rcpp::export(name = "BSF")]]
List BSF(arma::vec obs, int N, arma::rowvec param) {
  
  BS_Model filter(obs, N, param);
  arma::vec weights;
  int tmax = obs.n_elem;
  double log_marginal = 0.0;
  
  for (int t = 1; t < tmax + 1; ++t) {
    filter.transition(t);
    weights = filter.likelihood(t);
    log_marginal += log(arma::mean(weights));
    weights = normalize(weights);
    filter.resample(t, weights);
  }
  arma::mat states = rowMeans(filter.getSamples());
  arma::colvec s = filter.sample(weights);
  return List::create(Named("states") = states,
                      Named("sample") = s,
                      Named("log_marginal") = log_marginal);
}
