#include <Rcpp.h>
using namespace Rcpp;
#include <fstream>

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
    //std::ofstream outfile;
    //outfile.open("weights.txt", std::ios_base::app); // append instead of overwrite
    //outfile << "Range of NormWeights: " << (NumericVector)range(weightsnorm.row(t)); 
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


// ALPHA

// Sample theta proposal q(theta* | theta) In our case theta is beta. 
// [[Rcpp::export(name="param_proposal")]]
double qalpha(double thetagiven){
  //return Rcpp::rgamma(1, 10, thetagiven/10)[0]; 
  //R::rnorm(thetagiven, 0.5);
  return Rcpp::rbeta(1, 50, 50/thetagiven - 50)[0];
}

// Prior for theta. Evaluates prior density
// [[Rcpp::export(name="logp")]]
double logpalpha(double theta){
  return R::dbeta(theta, 6, 6/0.8 - 6, true); //R::dnorm(theta, 0.9, 0.5, true);
}

// Evaluates q(theta*|theta)
// [[Rcpp::export(name="logqeval")]]
double logqalphaeval(double thetastar, double thetagiven){
  return R::dbeta(thetastar, 50, 50/thetagiven - 50, true);   //R::dgamma(thetastar, 10, thetagiven/10, true); //R::dnorm(thetastar, thetagiven, 0.5, true);
}

 // SIGMA 

// transition sigma
// [[Rcpp::export(name="qsigma")]]
double qsigma(double thetagiven){
    return Rcpp::rgamma(1, 5, thetagiven/(double)5)[0];
}

// prior sigma
// [[Rcpp::export(name="priorsigma")]]
double logpsigma(double theta){
  return R::dgamma(theta, 5, 0.2, true);
}

// [[Rcpp::export(name="priorsigmatest")]]
double logpsigmatest(double theta, double shape, double rate){
  return R::dgamma(theta, shape, rate, true);
}

// evaluates transition sigma
// [[Rcpp::export(name="qsigmaeval")]]
double logqsigmaeval(double thetastar, double thetagiven){
  return R::dgamma(thetastar, 5, thetagiven/(double)5, true);
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
  bool inbound;
  double ap;   // acceptance probability
  
  // BURN IN
  for (int i=0; i < burnin; i++){
    // Sample theta* from q(theta* | theta)
    //std::cout << "write " << theta << " something" << std::endl;
    theta_candidate = qsigma(theta);
    //std::cout << "Cand: " << theta_candidate << std::endl;
    //inbound = (0.0001 < theta_candidate) && (theta_candidate < 0.9999) && (theta_candidate != 1);
    inbound = (0.0001 < theta_candidate) && (theta_candidate != 0);
    if (inbound){
      // Sample a candidate by running APF. Extract sample and log marginal
      //std::cout << "The eagle is in the nest, I repeat the eagle is in the nest." << std::endl;
      out = bf_cpp(y, N, beta, alpha, theta_candidate);
      x_candidate = out["sample"];
      logm_candidate = out["log_marginal"];
    }
    ap = logm_candidate + logpsigma(theta_candidate) - logm - logpsigma(theta) + logqsigmaeval(theta, theta_candidate) - logqsigmaeval(theta_candidate, theta) + log(inbound);
    std::cout << "iteration: " << i << " Theta candidate: " << theta_candidate << " P(cand): " << exp(logpsigma(theta_candidate)) << " P(curr): "<< exp(logpsigma(theta)) << " AP: "<< ap << std::endl;
    // Compute acceptance ratio
    if (logu[i] <= ap){
      // Accept!
      theta = theta_candidate;
      x = x_candidate;
      logm = logm_candidate;
    }
  }
  //std::cout << "burn in done" << std::endl;
  
  // MAIN LOOP
  for (int i=0; i < niter; i++){
    // Sample theta* from q(theta* | theta)
    theta_candidate = qalpha(theta);
    //std::cout << "Cand: " << theta_candidate << std::endl;
    // check 0 < < 1
    //inbound = (0.0001 < theta_candidate) && (theta_candidate < 0.9999) && (theta_candidate != 1);
    inbound = (0.0001 < theta_candidate) && (theta_candidate != 0);
    if (inbound){
      // Sample a candidate by running APF. Extract sample and log marginal
      //std::cout << "The eagle is in the nest, I repeat the eagle is in the nest." << std::endl;
      out = bf_cpp(y, N, beta, alpha, theta_candidate);
      x_candidate = out["sample"];
      logm_candidate = out["log_marginal"];
    }
    // Sample a candidate by running APF. Extract sample and log marginal
    std::cout << "iteration: " << i << " Theta candidate: " << theta_candidate << "P(cand): " << exp(logpsigma(theta_candidate)) << "P(curr): "<< exp(logpsigma(theta)) << " AP: "<< ap  << std::endl;
    //ap = logm_candidate + logpalpha(theta_candidate) - logm - logpalpha(theta) + logqalphaeval(theta, theta_candidate) - logqalphaeval(theta_candidate, theta) + log(inbound);
    ap = logm_candidate + logpsigma(theta_candidate) - logm - logpsigma(theta) + logqsigmaeval(theta, theta_candidate) - logqsigmaeval(theta_candidate, theta) + log(inbound);
    // Compute acceptance ratio
    if (logu[i] <= ap){
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



