// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
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
      this->_particles = arma::zeros(tmax, num_particles); 
      this->_obs = obs;
      this->_param = param;
      // Set x0:
      double sd = sqrt(param(2) / (1 - param(0)));
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
    
    arma::colvec filtered_states(arma::vec weights) {
      arma::colvec filtered(this->_particles.n_rows);
      int ncol = this->_particles.n_cols;
      for (int i = 0; i < ncol; ++i) {
        filtered = filtered + weights(i) * this->_particles.col(i);
      }
      return filtered;
    }
    
};

arma::vec elem_mult(arma::vec a, arma::vec b) {
  int n = a.n_elem;
  arma::vec out(n);
  for (int i = 0; i < n; ++i) {
    out(i) = a(i) * b(i);
  }
  return out;
}

arma::vec normalize(arma::vec v) {
  double total = arma::accu(v);
  return v / total;
}

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

Rcpp::List APF(arma::colvec obs, int num_particles, arma::vec param) {
  // INITIALISATION: (not sure if this is right...)
  SS_Model model(obs, num_particles, param);
  arma::vec weights = normalize(model.aux_likelihoodhood(1, obs(1)));
  
  // MAIN ITERATION:
  int tmax = obs.n_elem;
  arma::vec lik, aux_lik, parent_weights;
  arma::uvec resampled;
  double full_log_marginal = 0.0;
  for (int t = 1; t < tmax; ++t) {
    aux_lik = model.aux_likelihoodhood(t, obs(t));
    full_log_marginal += log_marginal(weights, aux_lik);
    parent_weights = normalize(elem_mult(weights, aux_lik));
    resampled = model.resample(t - 1, parent_weights);
    parent_weights = parent_weights.elem(resampled);
    weights = weights.elem(resampled);
    model.transition(t);
    lik = model.likelihood(t, obs(t));
    weights = update_weights(weights, parent_weights, lik);
    weights = normalize(weights);
  }
  arma::colvec sampled_particle = model.sample(weights);
  arma::colvec filtered = model.filtered_states(weights);
  return Rcpp::List::create(Rcpp::Named("sample") = sampled_particle,
                            Rcpp::Named("log_marginal") = full_log_marginal,
                            Rcpp::Named("filtered_states") = filtered);
}


//' Auxiliary Particle Filter
//' 
//' This function is currently limited to the stochastic volatility model.
//' 
//' @param obs A vector of observations to be filtered.
//' @param num_particles A positive integer number of particles to be used
//' in the simulation.
//' @param param A vector of model parameters.
//' 
//' @return A list containing a sample from the empirical distribution; the
//' approximated marginal log-likelihood of the data; the filtered states.
// [[Rcpp::export(name = "APF")]]
Rcpp::List APF_r(arma::colvec obs, int num_particles, arma::vec param) {
  return APF(obs, num_particles, param);
}