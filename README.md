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

* **Usage:** 
