# SeqInference

A repository containing group project work associated with the course:

Statistical Methods 2 - COMPASS, Bristol University.

Contributors: Andrea Becsek, Mauro Camara, Doug Corbin.

# Sequential Bayesian Parameter Inference

* *Description:* In this project you will use Sequential Monte Carlo (SMC) to recursively approximate a sequence of posterior distributions. In addition to allow to “see” the evolution of the posterior distribution as the sample size increases, an important advantage of SMC over Markov Chain Monte Carlo methods is that an estimate of the model evidence (on which Bayesian model choice relies) is
directly obtained as a by-product of the algorithm. 

* *Model:* Sequential Monte Carlo can be applied to approximate the posterior distribution of a parameter in a wide range of statistical models, including state-space models, models with latent variables and Gaussian process regression models. The choice of the model (and of the data) will depend on your own interests, but the model should lead to a non trivial implementation of SMC.

* *Main objectives:* Understand and implement SMC on a non trivial model.

# Repository Contents
## SMC Package

* **Installation:** Download `SMC_1.0.tar.gz` from the repository and run the command

```
R CMD INSTALL SMC_1.0.tar.gz
```

* **Usage:** A brief tutorial on the `SMC` package is given in `smc_package_tutorial.pdf`/`smc_package_tutorial.Rmd`.

* **Source Code:** Navigate to `SMC/src`.
 
  * `APF.cpp`: Source code associated with the Auxiliary Particle Filter.
  
  * `BSF.cpp`: Source code associated with the Bootstrap Particle Filter.
  
  * `metrop1.cpp`: Particle Metropolis Hasting implementation (with **beta/gamma** priors and **beta/gamma** proposal distirbutions).
  
  * `metrop2.cpp`: Particle Metropolis Hasting implementation (with **truncated-Normal/gamma** prior and **reflective random walk** proposal distirbutions).
  
  * `utils.cpp`: Useful generic functions.
  
  * `SMC.h`/`utils.h`: Header files for functions used accross multiple files within the package.
  
## Python

* `Pyhton/smc.py` contains an object oriented implementation of the Bootstrap Particle Filter.

## C++

 * `Cpp/bf_cpp_alpha.cpp` Source code of an Rcpp Bootstrap Filter with **beta** prior/transition for alpha, and **gamma** prior/transition for sigma. It also contains an implementation of `pmmh` in the log scale.
 
# Dirac Delta Appendix and Importance Sampling Plots
 * `DiracDelta_and_ImportanceSampling.Rmd` contains code to reproduce plots used in the appendix about Dirac Delta and to explain importance sampling.

  
