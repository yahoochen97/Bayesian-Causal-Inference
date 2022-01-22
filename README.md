# Bayesian Causal Inference

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Bayesian Causal Inference](#bayesian-causal-inference)
  - [Table of Contents](#table-of-contents)
  - [About The Project](#about-the-project)
  - [Model](#model)
    - [Built With](#built-with)
  - [Usage](#usage)
  - [License](#license)
  - [Contact](#contact)
  - [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project

A Bayesian alternative for Difference-in-difference causal model based on multi-task Gaussian Process regression.

## Model

Our model follows the classic setup of Diff-in-diff model. For units in both treatment and control group, we obtain time and unit dependent potentially noisy observation Y_it and covariates X_it. We assume for now that there are no interative effects in the data generation process.

```math
f(x,t,g) = h(x) + f_g(t) + u(t)+ e
```

where h(x) is the covariate effect, f(t) is the time trend for group g and u(t) is unit-specific trend. We place independent Gaussian Process priors on h and u but a joint GP on [$f_1$, $f_2$], so the outcome induces to another GP.

<!-- <img src="https://latex.codecogs.com/png.latex?f(x,t)\sim\mathcal{GP}(\mu_g(x)+\mu_h(t),K_g(x,x)+K_h(t,t)+\sigma^2I)" />  -->

<!-- The hyperparameters are optimized using the likelihood of all observations in control group and observations in treatment group until intervention. -->


### Built With
* [Python](https://www.python.org)
* [Pytorch](https://pytorch.org/)
* [Gpytorch](https://gpytorch.ai)
  


<!-- USAGE EXAMPLES -->
## Usage

The script localnews.m shows how to obtain MAP estimator for the multi-task GP model. Check load_data.m and localnewsmodel.m for how to customize the loading of data and specifying mean/covariance/prior function.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Yehu Chen - chenyehu@wustl.edu


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [gpytorch](https://gpytorch.ai)
* [GitHub Pages](https://pages.github.com)

