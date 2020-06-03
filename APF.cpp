// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
using namespace arma;

class SS_Model {
  private:
    //Attributes
    int _num_particles;
    mat _particles;
    vec _param;
    colvec _obs;
    
  public:
    //Constructor
    SS_Model(colvec obs, int num_particles, vec param) {
      this->_num_particles = num_particles;
      int tmax = obs.n_rows;
      this->_particles = zeros(tmax, num_particles); 
      this->_obs = obs;
      this->_param = param;
      // Set x0:
      double sd = sqrt(param(2) / (1 - param(0)));
      this->_particles.row(0) = Rcpp::as<rowvec>(Rcpp::rnorm(num_particles, 0, sd));
    }
    
    mat getParticles() {
      return this->_particles;
    }
    
    void transition(int t) {
      rowvec propagate(this->_num_particles);
      rowvec curr_particles = this->_particles.row(t - 1);
      double mean, sd;
      for (int i = 0; i < this->_num_particles; ++i) {
        mean = this->_param(0) * curr_particles(i);
        sd = sqrt(this->_param(2));
        propagate(i) = Rcpp::rnorm(1, mean, sd)[0];
      }
      this->_particles.row(t) = propagate;
    }
    
    uvec resample(int t, vec weights) {
      Rcpp::NumericVector to_sample, resampled, probs;
      to_sample = linspace(0, this->_num_particles - 1, this->_num_particles);
      probs = Rcpp::wrap(weights);
      resampled = Rcpp::sample(to_sample, this->_num_particles, true, probs);
      uvec resam = Rcpp::as<uvec>(resampled);
      this->_particles = this->_particles.cols(resam);
      return resam;
    }
    
    vec aux_likelihoodhood(int t, double obs) { 
      Rcpp::NumericVector obsv(1); obsv[0] = obs; // Convert obs to vector.
      rowvec curr_particles = this->_particles.row(t - 1);
      vec lik(this->_num_particles);
      double state_mean, lik_sd;
      for (int i = 0; i < this->_num_particles; ++i) {
        state_mean = this->_param(0) * curr_particles(i);
        lik_sd = this->_param(1) * exp(state_mean / 2);
        lik(i) = Rcpp::dnorm(obsv, 0.0, lik_sd)[0];
      }
      return lik;
    }
    
    vec likelihood(int t, double obs) {
      Rcpp::NumericVector obsv(1); obsv[0] = obs; // Convert obs to vector.
      rowvec curr_particles = this->_particles.row(t);
      vec lik(this->_num_particles);
      double lik_sd;
      for (int i = 0; i < this->_num_particles; ++i) {
        lik_sd = this->_param(1) * exp(curr_particles(i) / 2);
        lik(i) = Rcpp::dnorm(obsv, 0.0, lik_sd)[0];
      }
      return lik;
    }
    
    colvec filtered_states(vec weights) {
      colvec filtered(this->_particles.n_rows);
      for (int i = 0; i < this->_particles.n_cols; ++i) {
        filtered = filtered + weights(i) * this->_particles.col(i);
      }
      return filtered;
    }
    
    
    
};

vec elem_mult(vec a, vec b) {
  int n = a.n_elem;
  vec out(n);
  for (int i = 0; i < n; ++i) {
    out(i) = a(i) * b(i);
  }
  return out;
}

vec normalize(vec v) {
  double total = accu(v);
  return v / total;
}

vec update_weights(vec weights, vec parent_weights, vec lik) {
  int n = weights.n_elem;
  vec updated(n);
  for (int i = 0; i < n; ++i) {
    updated(i) = weights(i) / parent_weights(i) * lik(i);
  }
  return normalize(updated);
}

mat APF(colvec obs, int num_particles, vec param) {
  // INITIALISATION: (not sure if this is right...)
  SS_Model model(obs, num_particles, param);
  vec weights = normalize(model.aux_likelihoodhood(1, obs(1)));
  
  // MAIN ITERATION:
  int tmax = obs.n_elem;
  vec lik, aux_lik, parent_weights;
  uvec resampled;
  for (int t = 1; t < tmax; ++t) {
    aux_lik = model.aux_likelihoodhood(t, obs(t));
    parent_weights = normalize(elem_mult(weights, aux_lik));
    resampled = model.resample(t - 1, parent_weights);
    parent_weights = parent_weights.elem(resampled);
    weights = weights.elem(resampled);
    model.transition(t);
    lik = model.likelihood(t, obs(t));
    weights = update_weights(weights, parent_weights, lik);
    weights = normalize(weights);
  }
  return model.filtered_states(weights);
}

// [[Rcpp::export(name = "APF")]]
mat APF_r(colvec obs, int num_particles, vec param) {
  mat particles = APF(obs, num_particles, param);
  return particles;
}