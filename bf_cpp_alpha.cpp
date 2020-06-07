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
  NumericMatrix particles(tmax+1, N);         // Stores the N particles for all tmax steps
  NumericMatrix resampled(tmax+1, N);         // Temporary matrix storing the particles when resampling
  NumericMatrix posterior_sample(tmax+1, N);  // Sample from the final posterior. Used by PMMH.
  NumericMatrix logweights(tmax+1, N);        // N unnormalized logweights (one per particle) at each time step
  NumericMatrix weightsnorm(tmax+1, N);       // N *normalized* logweights (one per particle) at each time step
  IntegerVector ix(N);                        // Resampled indeces, one per particle.
  double log_marginal = 0.0;                  // log p(y | theta) used by PMMH
  double maxlogw, sumweights;                 // used for log-sum-exp trick and numerical stability when computing weights
  // First iteration comes from the prior distribution
  particles.row(0) = prior(N, alpha, sigma);
  // Main loop
  for (int t=1; t < (tmax+1); t++){
    // Sample from the prior and calculate (normalized) weights 
    particles.row(t) = transition(particles.row(t-1), alpha, sigma);
    logweights.row(t) = likelihood(y[t], particles.row(t), beta);
    // Log-Sum-Exp trick to find wait and likelihood
    maxlogw = max(logweights.row(t));
    logweights.row(t) = exp(logweights.row(t) - maxlogw);
    sumweights = sum(logweights.row(t));
    weightsnorm.row(t) = logweights.row(t) / sumweights;
    log_marginal += maxlogw + log(sumweights) - log(N);
    // Sample indices based on weights and use them to resample the columns of particle
    //std::cout <<"iter "<<t<< " hdifhdif" << (NumericVector)weightsnorm.row(t) << std::endl;
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

// Sample theta proposal q(theta* | theta) In our case theta is alpha. 
// [[Rcpp::export(name="param_proposal")]]
double qalpha(double thetagiven){
  return Rcpp::rbeta(1, 100, 100/thetagiven - 100)[0]; //prviously 50
}

// Prior for theta. Evaluates prior density
// [[Rcpp::export(name="logp")]]
double logpalpha(double theta){
  return R::dbeta(theta, 6, 6/0.8 - 6, true);
}

// Evaluates q(theta*|theta)
// [[Rcpp::export(name="logqeval")]]
double logqalphaeval(double thetastar, double thetagiven){
  return R::dbeta(thetastar, 50, 50/thetagiven - 50, true);
}

 // SIGMA 

// transition sigma
// [[Rcpp::export(name="qsigma")]]
double qsigma(double thetagiven){
    return Rcpp::rgamma(1, 200, thetagiven/(double)200)[0]; //300 worked well
}


//UNCOMMENT THIS
// prior sigma
// // [[Rcpp::export(name="priorsigma")]]
// double logpsigma(double theta){
//   return R::dgamma(theta, 5, 0.2, true);
// }

// [[Rcpp::export(name="priorsigma")]]
double logpsigma(double theta){
  return R::dgamma(theta, 5, 0.5/(double)5, true);
}

// evaluates transition sigma
// [[Rcpp::export(name="qsigmaeval")]]
double logqsigmaeval(double thetastar, double thetagiven){
  return R::dgamma(thetastar, 5, thetagiven/(double)5, true);
}


// [[Rcpp::export(name="pmmh_cpp_bf")]]
List pmmh(double thetastart, int niter, int N, NumericVector y, int burnin, double alphastart, double beta){  // N is the number of particles
  
  // INITIALIZATION: Run APF and grab sample & log marginal, then set starting param
  List out = bf_cpp(y, N, beta, alphastart, thetastart); // Run APF
  NumericVector x = out["filtered_states"];              // Grab a sample from posterior
  double logm = out["log_marginal"];                     // Grab log marginal
  double theta = thetastart;                             // Set initial parameter
  double alpha = alphastart;
  NumericVector logu = log(runif(N));                    // Generate log() of uniform random numbers
  NumericVector samples(niter);                          // Instantiate a vector of samples. This will be outputted
  NumericVector alphasamples(niter);
  int accepted = 0;                                      // Counts the number of times we accept
  int alphaaccepted = 0;
  NumericVector log_marginals(niter);                      // Store the log marginals. Can be used to evaluate performance of algorithm
  double theta_candidate;
  double alpha_candidate;
  double logm_candidate;
  NumericVector x_candidate;
  bool inbound;
  bool inboundalpha;
  double ap;   // acceptance probability
  
  // BURN IN
  for (int i=0; i < burnin; i++){
    // Sample theta* from q(theta* | theta)
    theta_candidate = qsigma(theta);
    alpha_candidate = qalpha(alpha);
    std::cout <<"it " << i << " sigma " << theta << " candidate " << theta_candidate << " alpha" << alpha << "cand" << alpha_candidate << std::endl;
    inbound = (0.0001 < theta_candidate) && (theta_candidate != 0);
    inboundalpha = (0.0001 < alpha_candidate) && (alpha_candidate < 0.9999) && (alpha_candidate != 1);
    if (inbound && inboundalpha){
      // Sample a candidate by running APF. Extract sample and log marginal
      out = bf_cpp(y, N, beta, alpha_candidate, theta_candidate);
      x_candidate = out["sample"];
      logm_candidate = out["log_marginal"];
    }
    ap = logm_candidate - logm
      + logpsigma(theta_candidate) - logpsigma(theta)  + logqsigmaeval(theta, theta_candidate) - logqsigmaeval(theta_candidate, theta) + log(inbound)
      + logpalpha(alpha_candidate) - logpalpha(alpha)  + logqalphaeval(alpha, alpha_candidate) - logqalphaeval(alpha_candidate, alpha) + log(inboundalpha);
    // Compute acceptance ratio
    if (logu[i] <= ap){
      // Accept!
      theta = theta_candidate;
      alpha = alpha_candidate;
      x = x_candidate;
      logm = logm_candidate;
    }
  }

  // MAIN LOOP
  for (int i=0; i < niter; i++){
    // Sample theta* from q(theta* | theta)
    theta_candidate = qsigma(theta);
    alpha_candidate = qalpha(alpha);
    std::cout <<"it " << i << " sigma " << theta << " candidate " << theta_candidate << " alpha" << alpha << "cand" << alpha_candidate << std::endl;
    inbound = (0.0001 < theta_candidate) && (theta_candidate != 0);
    inboundalpha = (0.0001 < alpha_candidate) && (alpha_candidate < 0.9999) && (alpha_candidate != 1);
    if (inbound && inboundalpha){
      // Sample a candidate by running APF. Extract sample and log marginal
      out = bf_cpp(y, N, beta, alpha_candidate, theta_candidate);
      x_candidate = out["sample"];
      logm_candidate = out["log_marginal"];
      
    }
    // Sample a candidate by running APF. Extract sample and log marginal
    ap = logm_candidate - logm
      + logpsigma(theta_candidate) - logpsigma(theta)  + logqsigmaeval(theta, theta_candidate) - logqsigmaeval(theta_candidate, theta) + log(inbound)
      + logpalpha(alpha_candidate) - logpalpha(alpha)  + logqalphaeval(alpha, alpha_candidate) - logqalphaeval(alpha_candidate, alpha) + log(inboundalpha);
      
    // Compute acceptance ratio
    if (logu[i] <= ap){
      // Accept!
      theta = theta_candidate;
      alpha = alpha_candidate;
      x = x_candidate;
      logm = logm_candidate;
      accepted++;
      alphaaccepted++;
    }
    // Now add the sample
    samples[i] = theta;
    alphasamples[i] =alpha;
    log_marginals[i] = logm;
  }
  // Return a named list
  return Rcpp::List::create(Rcpp::Named("acceptance") = accepted/(double)niter,
                            Rcpp::Named("acceptancealpha") = alphaaccepted/(double)niter,
                            Rcpp::Named("samples") = samples,
                            Rcpp::Named("alphasamples") = alphasamples,
                            Rcpp::Named("log_marginals") = log_marginals,
                            Rcpp::Named("final_states") = x);
}



