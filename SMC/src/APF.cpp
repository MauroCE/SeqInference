// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "utils.h"
using namespace Rcpp;

class SS_Model {
  private:
    //Attributes
    int _num_particles;
    arma::mat _particles;
    arma::vec _param;
    arma::colvec _obs;
    
  public:
    //Constructor
    SS_Model(arma::colvec obs, int num_particles, arma::vec param) {
      this->_num_particles = num_particles;
      int tmax = obs.n_rows;
      this->_particles = arma::zeros(tmax + 1, num_particles); 
      this->_obs = obs;
      this->_param = param;
      // Set x0:
      double sd = sqrt(pow(param(2), 2) / (1 - pow(param(0), 2)));
      this->_particles.row(0) = Rcpp::as<arma::rowvec>(Rcpp::rnorm(num_particles, 0, sd));
    }
    
    arma::mat getParticles() {
      return this->_particles;
    }
    
    void transition(int t) {
      arma::rowvec propagate(this->_num_particles);
      arma::rowvec curr_particles = this->_particles.row(t - 1);
      double mean, sd;
      for (int i = 0; i < this->_num_particles; ++i) {
        mean = this->_param(0) * curr_particles(i);
        sd = sqrt(this->_param(2));
        propagate(i) = Rcpp::rnorm(1, mean, sd)[0];
      }
      this->_particles.row(t) = propagate;
    }
    
    arma::colvec sample(arma::vec weights) {
      Rcpp::NumericVector to_sample, probs;
      to_sample = arma::linspace(0, this->_num_particles - 1, this->_num_particles);
      probs = Rcpp::wrap(weights);
      int sampled_index = Rcpp::sample(to_sample, 1, false, probs)[0];
      return this->_particles.col(sampled_index);
    }
    
    arma::uvec resample(int t, arma::vec weights) {
      Rcpp::NumericVector to_sample, resampled, probs;
      to_sample = arma::linspace(0, this->_num_particles - 1, this->_num_particles);
      probs = Rcpp::wrap(weights);
      resampled = Rcpp::sample(to_sample, this->_num_particles, true, probs);
      arma::uvec resam = Rcpp::as<arma::uvec>(resampled);
      this->_particles = this->_particles.cols(resam);
      return resam;
    }
    
    arma::vec aux_likelihoodhood(int t, double obs) { 
      Rcpp::NumericVector obsv(1); obsv[0] = obs; // Convert obs to vector.
      arma::rowvec curr_particles = this->_particles.row(t - 1);
      arma::vec lik(this->_num_particles);
      double state_mean, lik_sd;
      for (int i = 0; i < this->_num_particles; ++i) {
        state_mean = this->_param(0) * curr_particles(i);
        lik_sd = this->_param(1) * exp(state_mean / 2);
        lik(i) = Rcpp::dnorm(obsv, 0.0, lik_sd)[0];
      }
      return lik;
    }
    
    arma::vec likelihood(int t, double obs) {
      Rcpp::NumericVector obsv(1); obsv[0] = obs; // Convert obs to vector.
      arma::rowvec curr_particles = this->_particles.row(t);
      arma::vec lik(this->_num_particles);
      double lik_sd;
      for (int i = 0; i < this->_num_particles; ++i) {
        lik_sd = this->_param(1) * exp(curr_particles(i) / 2);
        lik(i) = Rcpp::dnorm(obsv, 0.0, lik_sd)[0];
      }
      return lik;
    }
};

arma::vec update_weights(arma::vec weights, arma::vec parent_weights, arma::vec lik) {
  int n = weights.n_elem;
  arma::vec updated(n);
  for (int i = 0; i < n; ++i) {
    updated(i) = weights(i) / parent_weights(i) * lik(i);
  }
  return normalize(updated);
}

double log_marginal(arma::vec weights, arma::vec aux_lik) { // Not the full marginal !!
  arma::vec weights_norm = normalize(weights);
  double weight_mean = arma::mean(weights);
  double aux_lik_mean = arma::accu(elem_mult(weights_norm, aux_lik));
  return log(weight_mean * aux_lik_mean);
}

//' Auxiliary Particle Filter
//' 
//' This function is currently limited to the stochastic volatility model.
//' 
//' @param obs A vector of observations to be filtered.
//' @param num_particles A positive integer number of particles to be used
//' in the simulation.
//' @param param A vector of model parameters (alpha, beta, sigma).
//' 
//' @return A list containing a sample from the empirical distribution; the
//' approximated marginal log-likelihood of the data; the sampled particles 
//' and their associated weights.
// [[Rcpp::export(name = "APF_Cpp")]]
Rcpp::List APF(arma::colvec obs, int num_particles, arma::vec param) {
  // INITIALISATION: (not sure if this is right...)
  SS_Model model(obs, num_particles, param);
  arma::vec weights = model.aux_likelihoodhood(1, obs(0));
  
  // MAIN ITERATION:
  int tmax = obs.n_elem;
  arma::vec lik, aux_lik, parent_weights;
  arma::uvec resampled;
  double full_log_marginal = 0.0;
  for (int t = 1; t < tmax + 1; ++t) {
    aux_lik = model.aux_likelihoodhood(t, obs(t - 1));
    full_log_marginal += log_marginal(weights, aux_lik);
    weights = normalize(weights);
    parent_weights = normalize(elem_mult(weights, aux_lik));
    resampled = model.resample(t - 1, parent_weights);
    parent_weights = parent_weights.elem(resampled);
    weights = weights.elem(resampled);
    model.transition(t);
    lik = model.likelihood(t, obs(t - 1));
    weights = update_weights(weights, parent_weights, lik);
  }
  arma::colvec sampled_particle = model.sample(weights);
  return Rcpp::List::create(Rcpp::Named("sample") = sampled_particle,
                            Rcpp::Named("log_marginal") = full_log_marginal,
                            Rcpp::Named("particles") = model.getParticles(),
                            Rcpp::Named("weights") = weights);
}
