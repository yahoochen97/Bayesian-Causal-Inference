# Bayesian Causal Inference

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Bayesian Causal Inference](#bayesian-causal-inference)
  - [Table of Contents](#table-of-contents)
  - [About The Project](#about-the-project)
  - [Model](#model)
  - [Example](#example)
    - [Built With](#built-with)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)
  - [Contact](#contact)
  - [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project

A Bayesian alternative for Difference-in-difference causal model based on multi-task Gaussian Process regression.

## Model

Our model follows the classic setup of Diff-in-diff model. For units in both treatment and control group, we obtain time and unit dependent potentially noisy observation Y_it and covariates X_it. We assume for now that there are no interative effects in the data generation process:

```math
f(x,t) = g(x) + h(t) + e
```

where g(x) is the covariate effect and h(t) is the time trend. We place two Gaussian Process priors on g and h, so the outcome induces to another GP:

```math
f(x,t) ~ GP(\mu_g(x)+\mu_h(t), K_g(x,x)+K_h(t,t)+\sigma^2 I)
```

The hyperparameters are optimized using the likelihood of all observations in control group and observations in treatment group until intervention.

## Example

```math
x_{it}=1+a*b+a+b+e
```

where there is one unobserved confounder a following N(0,1). The loadings are b_co ~ U[-1,1] and b_tr ~ U[-0.6, 1.4]. The error e follows $N(0,1)$.

```math
y_{it} = delta*D + \sum((2d+1)*x_{itd}) + \alpha_t + \beta + e
```

where the time trend $alpha_t$ = [sin(t) + t]/5 and group effects are $beta_co$ ~ U[-1,1] and $beta_tr$ ~ U[-0.6, 1.4]. The error e ~ N(0, 1).

### Built With
* [Python](https://www.python.org)
* [Pytorch](https://pytorch.org/)
* [Gpytorch](https://gpytorch.ai)
  

<!-- GETTING STARTED -->
## Getting Started

To run this project, users need to have python, pytorch and gpytorch installed locally.

### Prerequisites


### Installation


<!-- USAGE EXAMPLES -->
## Usage



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

