#ifndef __SMC__
#define __SMC__

Rcpp::List APF(arma::colvec obs, int num_particles, arma::vec param);
Rcpp::List BSF(arma::vec obs, int N, arma::rowvec param);

#endif // __SMC __