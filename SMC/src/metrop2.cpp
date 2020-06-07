// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "SMC.h"
#include "utils.h"
using namespace Rcpp;

class Metrop_Model {
  private:
    // ATTRIBUTES:
    arma::mat _pchain; // Chain for parameters.
    arma::mat _xchain; // Chain for states.
    arma::mat _limits; // Bounds for parameters.
    int _tmax; // Number of iterations.
    double _sd; // SD of proposal steps.
  
  public:
    // CONSTRUCTOR:
    Metrop_Model(int tmax, int N, arma::rowvec initial, double sd) {
      this->_tmax = tmax;
      this->_pchain = arma::zeros(tmax + 1, initial.n_elem);
      this->_pchain.row(0) = initial;
      this->_xchain = arma::zeros(tmax, N);
      this->_sd = sd;
      arma::mat lim = arma::zeros(initial.n_elem, 2);
      // Alpha:
      lim(0,0) = -1;
      lim(0,1) = 1;
      // Beta:
      lim(1,0) = 0;
      lim(1,1) = arma::datum::inf;
      // Sigma:
      lim.row(2) = lim.row(1);
      this->_limits = lim;
    }
    
    void propose(int t) {
      int nparam = this->_pchain.n_cols;
      arma::rowvec param0 = this->_pchain.row(t - 1);
      arma::rowvec param1(nparam);
      double prop;
      for (int i = 0; i < nparam; ++i) {
        prop = rnorm(1, param0(i), this->_sd)[0];
        param1(i) = reflect(param0(i), prop, this->_limits.row(i));
      }
      this->_pchain.row(t) = param1;
    }
    
    double prior(arma::rowvec param) {
      // Beta and Sigma follow gamma distribution.
      double out = 0.0;
      NumericVector beta(1);
      NumericVector sigma(1);
      beta[0] = param(1);
      sigma[0] = param(2);
      out += dgamma(beta, 5, 10, true)[0] + dgamma(sigma, 20, 20, true)[0];
      
      // Alpha follows truncated normal prior.
      double alpha = param(0);
      out += log(trunc_dnorm(alpha, 0.9, 0.05, this->_limits.row(0)));
      return out;
    }
    
    double qeval(int t1, int t0) {
      arma::rowvec param0 = this->_pchain.row(t0);
      arma::rowvec param1 = this->_pchain.row(t1);
      double out = 0.0;
      NumericVector v(1);
      int nparam = param0.n_elem;
      for (int i = 0; i < nparam; ++i) { // UNSIGNED INTEGER ERROR
        v[0] = param1(i);
        out += dnorm(v, param0(i), this->_sd)[0];
      }
      return out;
    }
    
    void reject(int t) {
      this->_pchain.row(t) = this->_pchain.row(t - 1); 
    }
    
    // Attribute access...
    arma::mat getPchain() {
      return this->_pchain;
    }
    
    arma::rowvec getParam(int t) {
      return this->_pchain.row(t);
    }
};

arma::mat pseudo_mh(int tmax, arma::vec obs, int N, arma::rowvec initial, double sd) {
  
  // INITIALIZE:
  Metrop_Model mh(tmax, N, initial, sd);
  List result = BSF(obs, N, initial);
  double lmarg0 = as<double>(result["log_marginal"]);
  
  arma::rowvec p0, p1;
  double lmarg1, prior0, prior1, numer, denom, prob;
  NumericVector u = runif(tmax, 0, 1);
  int accept = 0;
  
  // MAIN:
  for (int t = 1; t < tmax + 1; ++t) {
    Rcout << t << std::endl;
    mh.propose(t);
    p0 = mh.getParam(t - 1);
    p1 = mh.getParam(t);
    result = BSF(obs, N, p1);
    prior0 = mh.prior(p0);
    prior1 = mh.prior(p1);
    lmarg1 = as<double>(result["log_marginal"]);
    numer = lmarg1 + prior1;
    denom = lmarg0 + prior0;
    prob = numer - denom;
    if (log(u[t - 1]) <= prob) {
      accept += 1;
      continue;
    } else {
      mh.reject(t);
    }
  }
  Rcout << "Acceptance: " << accept << std::endl;
  return mh.getPchain();
}