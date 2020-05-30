#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadilloExtensions/sample.h>
//#include <cmath>
  using namespace Rcpp;

// [[Rcpp::export]]
arma::vec vec_dnorm(double x, double means, arma::vec sd) {
  int n = sd.size();
  arma::vec res(n);
  for(int i = 0; i < n; i++) {
    res[i] = R::dnorm(x, means, sd[i], FALSE);
  }
  return res;
}


//[[Rcpp::export(name = "ap_filter_sv")]]
arma::mat particle_filter(arma::vec& y,
                          arma::vec& theta,
                          int& n_particles){
  
  int T = y.size() - 1;
  double alpha = theta[0];
  double beta = theta[1];
  double sigma = theta[2];
  
  //INITIALIZE VARIABLES
  arma::mat particles = arma::randu<arma::mat>(n_particles, T+1);
  //arma::mat indeces = arma::randu<arma::mat>(n_particles, T+1);
  //arma::mat weights = arma::randu<arma::mat>(n_particles, T+1);
  //arma::mat normalized_weights = arma::randu<arma::mat>(n_particles, T+1);
  arma::mat x_hat_filtered = arma::randu<arma::vec>(T);
  arma::mat weights_1 = arma::randu<arma::vec>(n_particles);
  arma::mat weights_1_norm = arma::randu<arma::vec>(n_particles);
  arma::mat weights_2 = arma::randu<arma::vec>(n_particles);
  arma::mat weights_2_norm = arma::randu<arma::vec>(n_particles);
  
  double log_likelihood = 0;
  
  //INITIALIZE STATE
   NumericVector initial_particles(n_particles);
   initial_particles = rnorm(n_particles, 0, sqrt(pow(sigma,2)/(1-pow(alpha,2))));
   particles.col(0) = as<arma::Col<double>> (initial_particles);
     
  //particles.col(0) = init_state * arma::ones<arma::vec>(n_particles);
   x_hat_filtered(0) = mean(particles.col(0));
  //normalized_weights.col(0) = (1 / double(n_particles))*arma::randu<arma::vec>(n_particles);
  
  for(int t=1; t<T; t++){
    
    //RESAMPLE USING MULTINOMIAL

    weights_1 = vec_dnorm(y(t), 0, beta * arma::exp(alpha*particles.col(t-1)/2));
    double weights_1_sum = accu(weights_1);
    weights_1_norm = weights_1 / weights_1_sum;
    NumericVector resampled_particles = Rcpp::sample(as<NumericVector>(wrap(particles.col(t-1))), 
                                                     n_particles, 
                                                     true, 
                                                     as<NumericVector>(wrap(weights_1_norm)));
    particles.col(t-1) = as<arma::Col<double>> (resampled_particles);
    
    //PROPAGATE
    NumericVector noise(n_particles);
    noise = sigma * rnorm(n_particles,0,1);
    
    NumericVector new_particles(n_particles);
    new_particles= alpha * particles.col(t-1) + as<arma::Col<double>> (noise);
    particles.col(t) = as<arma::Col<double>>(new_particles);
    
    //COMPUTE NEW WEIGHTS
    weights_2 = vec_dnorm(y(t), 0, beta * arma::exp(alpha*particles.col(t)/2));
    double weights_2_sum = accu(weights_2);
    weights_2_norm = weights_2 / weights_2_sum;
    
    //RESAMPLE
    resampled_particles = Rcpp::sample(as<NumericVector>(wrap(particles.col(t))), 
                                       n_particles, 
                                       true, 
                                       as<NumericVector>(wrap(weights_2_norm)));
    particles.col(t) = as<arma::Col<double>> (resampled_particles);
    
    //ESTIMATE STATE
    x_hat_filtered(t) = mean(particles.col(t));
    
  }
  return x_hat_filtered;
}
