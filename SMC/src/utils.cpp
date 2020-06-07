// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

arma::vec elem_mult(arma::vec a, arma::vec b) {
  int n = a.n_elem;
  arma::vec out(n);
  for (int i = 0; i < n; ++i) {
    out(i) = a(i) * b(i);
  }
  return out;
}

arma::vec normalize(arma::vec v) {
  double total = arma::accu(v);
  return v / total;
}

arma::colvec rowMeans(arma::mat matrix) {
  int nrow = matrix.n_rows;
  arma::colvec out(nrow);
  for (int i = 0; i < nrow; ++i) {
    out(i) = arma::mean(matrix.row(i));
  }
  return out;
}

bool inside(double x, arma::rowvec limits) {
  if ((x > limits(0)) && (x < limits(1))) {
    return true;
  } else {
    return false;
  }
}

double reflect(double start, double end, arma::rowvec limits) {
  while (not inside(end, limits)) {
    int d = arma::sign(end - start);
    start = end; // Incase we reflect more than once.
    if (d == 1) { // Positive direction.
      end = limits(1) - (end - limits(1));
    } else {
      end = limits(0) + (limits(0) - end);
    }
  }
  return end;
}

double trunc_dnorm(double x, double mean, double sd, arma::rowvec limits) {
  
  NumericVector theta1(1); 
  NumericVector theta2(1); 
  NumericVector theta3(1);
  
  theta1[0] = (x - mean) / sd;
  theta2[0] = (limits(0) - mean) / sd;
  theta3[0] = (limits(1) - mean) / sd;
  
  return (1 / sd) * dnorm(theta1)[0] / (pnorm(theta3)[0] - pnorm(theta2)[0]);
}

arma::uvec which_nonzero(arma::vec v) {
  int n = v.n_elem;
  int count = 0;
  arma::uvec out(n);
  for (int i = 0; i < n; ++i) {
    if (v(i) >= arma::datum::eps) {
      out(count) = i;
      count += 1;  
    } else {
      continue;
    }
  }
  arma::vec index = arma::linspace(0, count - 1, count);
  arma::uvec index_out(count);
  for (int i = 0; i < count; ++i) {
    index_out(i) = index(i);
  }
  return out.elem(index_out);
}
