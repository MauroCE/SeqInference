#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export(name="pr")]]
NumericVector prior(int N, double alpha, double sigma){
  return rnorm(N, 0.0, pow(sigma,2) / (1 - pow(alpha,2)));
}

// [[Rcpp::export(name="lk")]]
NumericVector likelihood(double y, NumericVector x, double beta){
  int n = x.size();
  NumericVector like(n);
  for (int i=0; i < n; i++){
    like[i] = R::dnorm(y, 0.0, beta*exp(x[i]/2), true);
  }
  return like;
}

// [[Rcpp::export(name="tr")]]
NumericVector transition(NumericVector x, double alpha, double sigma){
  return alpha*x + rnorm(x.size(), 0.0, sigma); //rnorm(x.size(), 0.91*x, 1.0);
}

// [[Rcpp::export(name="bf_cpp")]]
List bf_cpp(NumericVector y, int N, double beta, double alpha, double sigma) {
  int tmax = y.size();
  // Initialize particles, resampled particles and weights
  NumericMatrix particles(tmax+1, N);
  NumericMatrix resampled(tmax+1, N);
  NumericMatrix posterior_sample(tmax+1, N);  // Sample from the final posterior. Used by PMMH
  NumericMatrix logweights(tmax+1, N);
  NumericMatrix weightsnorm(tmax+1, N);
  IntegerVector ix(N);
  double log_marginal = 0.0;
  double maxlogw, sumweights;
  // First iteration
  particles.row(0) = prior(N, alpha, sigma);
  //std::cout << "prior " << (NumericVector)particles.row(0) << std::endl;
  //std::cout << "prior func " << prior(N, alpha, sigma) << std::endl;
  // Main loop
  for (int t=1; t < (tmax+1); t++){
    // Sample from the prior and calculate (normalized) weights 
    //std::cout << "iteration " << t << std::endl;
    particles.row(t) = transition(particles.row(t-1), alpha, sigma);
    //std::cout << "parts " << (NumericVector)particles.row(t) << std::endl;
    logweights.row(t) = likelihood(y[t], particles.row(t), beta);
    //std::cout << "logweights" << (NumericVector)logweights.row(t) << " " << std::endl;
    // log sum exp to find wait and likelihood
    maxlogw = max(logweights.row(t));
    logweights.row(t) = exp(logweights.row(t) - maxlogw);
    sumweights = sum(logweights.row(t));
    weightsnorm.row(t) = logweights.row(t) / sumweights;
    log_marginal += maxlogw + log(sumweights) - log(N);
    // Sample indices based on weights and use them to resample the columns of particle
    //std::cout << "it " << weightsnorm << " it" << std::endl;
    ix = sample(N, N, true, (NumericVector)weightsnorm.row(t));
    for (int j=0; j < N; j++){
      resampled.column(j) = particles.column(ix[j]-1);
    }
    particles = resampled;
  }
  // Sample one last time from the particles
  ix = sample(N, N, true, (NumericVector)weightsnorm.row(tmax));
  for (int j=0; j < N; j++){
    posterior_sample.column(j) = particles.column(ix[j]-1);
  }
  // Return particles, posterior sample, 
  return Rcpp::List::create(Rcpp::Named("filtered_states") = rowMeans(particles),
                            Rcpp::Named("sample") = posterior_sample,
                            Rcpp::Named("log_marginal") = log_marginal);
}




// Sample theta proposal q(theta* | theta) In our case theta is beta. 
// [[Rcpp::export(name="param_proposal")]]
double q(double thetagiven){
  return Rcpp::rgamma(1, 10, thetagiven/10)[0]; //R::rnorm(thetagiven, 0.5);
}

// Prior for theta. Evaluates prior density
// [[Rcpp::export(name="logp")]]
double logp(double theta){
  return R::dgamma(theta, 10, 1/10, true); //R::dnorm(theta, 0.9, 0.5, true);
}

// Evaluates q(theta*|theta)
// [[Rcpp::export(name="logqeval")]]
double logqeval(double thetastar, double thetagiven){
  return R::dgamma(thetastar, 10, thetagiven/10, true); //R::dnorm(thetastar, thetagiven, 0.5, true);
}

// [[Rcpp::export(name="pmmh_cpp_bf")]]
List pmmh(double thetastart, int niter, int N, NumericVector y, int burnin, double alpha, double beta){  // N is the number of particles

  // INITIALIZATION: Run APF and grab sample & log marginal, then set starting param
  List out = bf_cpp(y, N, beta, alpha, thetastart); // Run APF
  NumericVector x = out["filtered_states"];             // Grab a sample from posterior
  double logm = out["log_marginal"];  // Grab log marginal
  double theta = thetastart;           // Set initial parameter
  NumericVector logu = log(runif(N));             // Generate log() of uniform random numbers
  NumericVector samples(niter);                // Instantiate a vector of samples. This will be outputted
  int accepted = 0;                 // Counts the number of times we accept
  NumericVector log_marginals(niter);         // Store the log marginals. Can be used to evaluate performance of algorithm
  double theta_candidate;
  double logm_candidate;
  NumericVector x_candidate;
  
  // BURN IN
  for (int i=0; i < burnin; i++){
    // Sample theta* from q(theta* | theta)
    //std::cout << "write " << theta << " something" << std::endl;
    theta_candidate = q(theta);
    // Sample a candidate by running APF. Extract sample and log marginal
    //std::cout << theta_candidate << std::endl;
    out = bf_cpp(y, N, beta, alpha, theta_candidate);
    x_candidate = out["sample"];
    logm_candidate = out["log_marginal"];
    // Compute acceptance ratio
    if (logu[i] <= logm_candidate + logp(theta_candidate) - logm - logp(theta) + logqeval(theta, theta_candidate) - logqeval(theta_candidate, theta)){
      // Accept!
      theta = theta_candidate;
      x = x_candidate;
      logm = logm_candidate;
    }
  }
  
  // MAIN LOOP
  for (int i=0; i < niter; i++){
    // Sample theta* from q(theta* | theta)
    theta_candidate = q(theta);
    // Sample a candidate by running APF. Extract sample and log marginal
    
    out = bf_cpp(y, N, beta, alpha, theta_candidate);
    x_candidate = out["sample"];
    logm_candidate = out["log_marginal"];
    // Compute acceptance ratio
    if (logu[i] <= logm_candidate + logp(theta_candidate) - logm - logp(theta) + logqeval(theta, theta_candidate) - logqeval(theta_candidate, theta)){
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



