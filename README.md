PYTHON LABELED RFS 
========================
This repository contains Python implementation of `jointglmb` GLMB [1] and `jointlmb` LMB [3]. The implementation are ported from `rfs_tracking_toolbox\jointlmb\gms` and `rfs_tracking_toolbox\jointglmb\gms` implemented in Matlab (it was done by Prof. Vo's research group). 
GLMB is originally, theoretically proposed in [0].

- A detail of how to implement Delta-GLMB (with two separated prediction and update steps) is given [2].   
- Adaptive birth is implemented based on [3] (Section __Adaptive Birth Distribution__), mainly focused on equation (75).  
- Sampling solutions (ranked assignments), `gibbs_multisensor_approx_cheap` is implemented in C++ based on __Algorithm 2: MM-Gibbs (Suboptimal)__ [4].  
- Adaptive birth is implemented based on [5] (implemented in C++ based on __Algorithm 1 Multi-sensor Adaptive Birth Gibbs Sampler__), Gaussian Likelihoods.

[0] Vo, Ba-Tuong, and Ba-Ngu Vo. "Labeled random finite sets and multi-object conjugate priors." IEEE Transactions on Signal Processing 61, no. 13 (2013): 3460-3475.  
[1] Vo, Ba-Ngu, Ba-Tuong Vo, and Hung Gia Hoang. "An efficient implementation of the generalized labeled multi-Bernoulli filter." IEEE Transactions on Signal Processing 65, no. 8 (2016): 1975-1987.  
[2] Vo, Ba-Ngu, Ba-Tuong Vo, and Dinh Phung. "Labeled random finite sets and the Bayes multi-target tracking filter." IEEE Transactions on Signal Processing 62, no. 24 (2014): 6554-6567.  
[3] Reuter, Stephan, Ba-Tuong Vo, Ba-Ngu Vo, and Klaus Dietmayer. "The labeled multi-Bernoulli filter." IEEE Transactions on Signal Processing 62, no. 12 (2014): 3246-3260.   
[4] Vo, B. N., Vo, B. T., & Beard, M. (2019). Multi-sensor multi-object tracking with the generalized labeled multi-Bernoulli filter. IEEE Transactions on Signal Processing, 67(23), 5952-5967.      
[5] Trezza, A., Bucci Jr, D. J., & Varshney, P. K. (2021). Multi-sensor Joint Adaptive Birth Sampler for Labeled Random Finite Set Tracking. arXiv preprint arXiv:2109.04355.
  
USAGE
=====
GLMB
* Original Matlab source: `jointglmb_gms_matlab`
* Original Python porting: `jointglmb_gms_python`
* Improved version (code optimized, adaptive birth): `jointglmb_gms_python_fast`

LMB
* Original Matlab source: `jointlmb_gms_matlab`
* Original Python porting: `jointlmb_gms_python`
* Improved version (code optimized): `jointlmb_gms_python_fast`
* No adaptive birth is implemented for simplification (but can be implemented similar to jointglmb)

Gibbs Sampling
* Python package for an efficient algorithm for truncating the GLMB filtering density based on Gibbs sampling.
* The implementation is done in C++ and based on __Algorithm 1. Gibbs__ (and _"Algorithm 1a"_) of paper [1].
* Python wrapper for faster computation

MS-GLMB
* `gibbs_multisensor_approx_cheap` is implemented in C++.  
* Adaptive birth is implemented in C++, Gaussian Likelihoods.

