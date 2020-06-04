#include <Rcpp.h>
using namespace Rcpp;



NumericVector prior(int N){
  return rnorm(N, 0.0, 1.0 / (1 - 0.8281));
}

// [[Rcpp::export(name="some")]]
NumericVector likelihood(double y, NumericVector x, double beta){
  int n = x.size();
  NumericVector like(n);
  for (int i=0; i < n; i++){
    like[i] = R::dnorm(y, 0.0, beta*exp(x[i]/2), false);
  }
  return like;
}

NumericVector transition(NumericVector x){
  return 0.91*x + rnorm(x.size(), 0.0, 1.0); //rnorm(x.size(), 0.91*x, 1.0);
}

// [[Rcpp::export(name="bf_cpp")]]
List bf_cpp(NumericVector y, int N, double beta) {
  int tmax = y.size();
  // Initialize particles, resampled particles and weights
  NumericMatrix particles(tmax+1, N);
  NumericMatrix resampled(tmax+1, N);
  NumericMatrix posterior_sample(tmax+1, N);  // Sample from the final posterior. Used by PMMH
  NumericVector weights(N);
  IntegerVector ix(N);
  double log_marginal = 0.0;
  // First iteration
  particles.row(0) = prior(N);
  // Main loop
  for (int t=1; t < (tmax+1); t++){
    // Sample from the prior and calculate (normalized) weights 
    particles.row(t) = transition(particles.row(t-1));
    weights = likelihood(y[t], particles.row(t), beta);
    // Use likelihood to compute the marginal likelihood
    log_marginal += log(mean(weights));
    // Normalize weights
    weights = weights / sum(weights);
    // Sample indices based on weights and use them to resample the columns of particle
    ix = sample(N, N, true, weights);
    for (int j=0; j < N; j++){
      resampled.column(j) = particles.column(ix[j]-1);
    }
    particles = resampled;
  }
  // Sample one last time from the particles
  ix = sample(N, N, true, weights);
  for (int j=0; j < N; j++){
    posterior_sample.column(j) = particles.column(ix[j]-1);
  }
  // Return particles, posterior sample, 
  return Rcpp::List::create(Rcpp::Named("filtered_states") = rowMeans(particles),
                            Rcpp::Named("sample") = posterior_sample,
                            Rcpp::Named("log_marginal") = log_marginal);
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

// [[Rcpp::export(name="pmmh_cpp_bf")]]
List pmmh(double thetastart, int niter, int N, NumericVector y) {  // N is the number of particles

  // INITIALIZATION: Run APF and grab sample & log marginal, then set starting param
  List out = bf_cpp(y, N, thetastart); // Run APF
  NumericVector x = out["sample"];             // Grab a sample from posterior
  double logm = out["log_marginal"];  // Grab log marginal
  double theta = thetastart;           // Set initial parameter
  NumericVector logu = log(runif(N));             // Generate log() of uniform random numbers
  NumericVector samples(niter);                // Instantiate a vector of samples. This will be outputted
  int accepted = 0;                 // Counts the number of times we accept
  NumericVector log_marginals(niter);         // Store the log marginals. Can be used to evaluate performance of algorithm
  double theta_candidate;
  double logm_candidate;
  NumericVector x_candidate;
  
  // MAIN LOOP
  for (int i=0; i < niter; i++){
    // Sample theta* from q(theta* | theta)
    theta_candidate = q(theta);
    // Sample a candidate by running APF. Extract sample and log marginal
    out = bf_cpp(y, N, theta_candidate);
    x_candidate = out["sample"];
    logm_candidate = out["log_marginal"];
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



