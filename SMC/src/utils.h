#ifndef __UTILS__
#define __UTILS__

arma::vec elem_mult(arma::vec a, arma::vec b);
arma::vec normalize(arma::vec v);
arma::colvec rowMeans(arma::mat matrix);
double reflect(double start, double end, arma::rowvec limits);
double trunc_dnorm(double x, double mean, double sd, arma::rowvec limits);
arma::uvec which_nonzero(arma::vec v);

#endif // __UTILS__