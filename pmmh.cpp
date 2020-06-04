// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
//#include <SMC/src/SMC.h>
//#include "SMC.h"
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
      mean = sqrt(this->_param(0)) * curr_particles(i);
      sd = sqrt(this->_param(2));
      propagate(i) = Rcpp::rnorm(1, mean, sd)[0];
    }
    this->_particles.row(t) = propagate;
  }
  
  colvec sample(vec weights) {
    Rcpp::NumericVector to_sample, probs;
    to_sample = linspace(0, this->_num_particles - 1, this->_num_particles);
    probs = Rcpp::wrap(weights);
    int sampled_index = Rcpp::sample(to_sample, 1, false, probs)[0];
    return this->_particles.col(sampled_index);
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
      state_mean = sqrt(this->_param(0)) * curr_particles(i);
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

double log_marginal(vec weights, vec aux_lik) { // Not the full marginal !!
  vec weights_norm = normalize(weights);
  double weight_mean = mean(weights);
  double aux_lik_mean = accu(elem_mult(weights_norm, aux_lik));
  return log(weight_mean * aux_lik_mean);
}

Rcpp::List APF(colvec obs, int num_particles, vec param) {
  // INITIALISATION: (not sure if this is right...)
  SS_Model model(obs, num_particles, param);
  vec weights = normalize(model.aux_likelihoodhood(1, obs(1)));
  
  // MAIN ITERATION:
  int tmax = obs.n_elem;
  vec lik, aux_lik, parent_weights;
  uvec resampled;
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
  colvec sampled_particle = model.sample(weights);
  colvec filtered = model.filtered_states(weights);
  return Rcpp::List::create(Rcpp::Named("sample") = sampled_particle,
                            Rcpp::Named("log_marginal") = full_log_marginal,
                            Rcpp::Named("filtered_states") = filtered);
}

// [[Rcpp::export(name = "APF")]]
Rcpp::List APF_r(colvec obs, int num_particles, vec param) {
  return APF(obs, num_particles, param);
}

// Sample theta propposal q(theta* | theta) In our case theta is beta. 
// [[Rcpp::export(name="q")]]
double q(double thetagiven){
  return R::rnorm(thetagiven, 0.1);
}

// Prior for theta. Evaluates prior density
// [[Rcpp::export(name="logp")]]
double logp(double theta){
  return R::dnorm(theta, 1.0, 0.2, true);
}

// Evaluates q(theta*|theta)
// [[Rcpp::export(name="logqeval")]]
double logqeval(double thetastar, double thetagiven){
  return R::dnorm(thetastar, thetagiven, 0.1, true);
}

// [[Rcpp::export(name="pmmh_cpp")]]
Rcpp::List pmmh(double thetastart, int niter, int N, vec y) {  // N is the number of particles
  vec param = {0.8281, thetastart, 1.0};
  
  // INITIALIZATION: Run APF and grab sample & log marginal, then set starting param
  Rcpp::List out = APF(y, N, param); // Run APF
  vec x = out["sample"];             // Grab a sample from posterior
  double logm = out["log_marginal"];  // Grab log marginal
  double theta = thetastart;           // Set initial parameter
  vec logu = log(Rcpp::runif(N));             // Generate log() of uniform random numbers
  vec samples(niter);                // Instantiate a vector of samples. This will be outputted
  int accepted = 0;                 // Counts the number of times we accept
  vec log_marginals(niter);         // Store the log marginals. Can be used to evaluate performance of algorithm
  double theta_candidate;
  double logm_candidate;
  vec x_candidate;
  
  // MAIN LOOP
  for (int i=0; i < niter; i++){
    // Sample theta* from q(theta* | theta)
    theta_candidate = q(theta);
    param = {0.8281, theta_candidate, 1.0};
    // Sample a candidate by running APF. Extract sample and log marginal
    out = APF(y, N, param);
    x_candidate = Rcpp::as<Col<double>>(out["sample"]);
    logm_candidate = Rcpp::as<double>(out["log_marginal"]);
    // Compute acceptance ratio
    if (logu[i] <= logm_candidate + logp(theta_candidate) - logm - logp(theta)){
      // Accept!
      theta = theta_candidate;
      x = x_candidate;
      logm = logm_candidate;
      accepted++;
    }
    // Now add the sample
    samples[i] = theta;
    log_marginals[i] = logm;
  }
  // Return a named list
  return Rcpp::List::create(Rcpp::Named("acceptance") = accepted/(double)niter,
                            Rcpp::Named("samples") = samples,
                            Rcpp::Named("log_marginals") = log_marginals);
}


