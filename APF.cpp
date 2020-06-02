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
      rowvec curr_particles = this->_particles.row(t);
      double mean, sd;
      for (int i = 0; i < this->_num_particles; ++i) {
        mean = this->_param(0) * curr_particles(i);
        sd = sqrt(this->_param(2));
        propagate(i) = Rcpp::rnorm(1, mean, sd)[0];
      }
      this->_particles.row(t + 1) = propagate;
    }
    
    void resample(int t, vec weights) {
      Rcpp::NumericVector to_sample, resampled, probs;
      to_sample = Rcpp::wrap(this->_particles.row(t));
      probs = Rcpp::wrap(weights);
      resampled = Rcpp::sample(to_sample, this->_num_particles, true, probs);
      arma::rowvec resam = Rcpp::as<rowvec>(resampled);
      this->_particles.row(t) = resam;
    }
    
    vec particle_likelihood(int t, double obs) { 
      Rcpp::NumericVector obsv(1); obsv[0] = obs;
      rowvec curr_particles = this->_particles.row(t);
      vec lik(this->_num_particles);
      double mean, state_mean, lik_sd;
      for (int i = 0; i < this->_num_particles; ++i) {
        state_mean = this->_param(0) * curr_particles(i);
        lik_sd = this->_param(1) * exp(state_mean / 2);
        lik(i) = Rcpp::dnorm(obsv, 0.0, lik_sd)[0];
      }
      return lik;
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

vec update_weights(vec weights, vec parent_weights, vec lik) {
  int n = weights.n_elem;
  vec updated(n);
  for (int i = 0; i < n; ++i) {
    updated(i) = weights(i) / parent_weights(i) * lik(i);
  }
  return normalise(updated);
}

mat APF(colvec obs, int num_particles, vec param) {
  // INITIALISATION:
  SS_Model model(obs, num_particles, param);
  vec weights = ones(num_particles); 
  weights = weights / weights.n_elem;
  vec lik = model.particle_likelihood(0, obs(0));
  vec parent_weights = normalise(elem_mult(weights, lik));
  model.resample(0, parent_weights);
  weights = update_weights(weights, parent_weights, lik);
  
  // MAIN ITERATION:
  int tmax = obs.n_elem;
  for (int t = 1; t < tmax; ++t) {
    model.transition(t - 1);
    lik = model.particle_likelihood(t, obs(t));
    Rcpp::Rcout << "lik: " << lik << std::endl;
    parent_weights = normalise(elem_mult(weights, lik));
    Rcpp::Rcout << "parent: " << parent_weights << std::endl;
    model.resample(t, parent_weights);
    weights = update_weights(weights, parent_weights, lik);
    Rcpp::Rcout << "weights: " << weights << std::endl;
  }
  return model.getParticles();
}

// [[Rcpp::export(name = "APF")]]
mat APF_r(colvec obs, int num_particles, vec param) {
  mat particles = APF(obs, num_particles, param);
  return particles;
}