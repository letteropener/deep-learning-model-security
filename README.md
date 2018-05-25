# Combined Adversarial Training

Train a model with adversarial samples from multiple attack types.<br>

<b>Goals:</b><br><br>

1)Observe whether this improves robustness to all these attacks (higher accuracy on adversarial examples)<br>
2)Still maintain good accuracy on legitimate data<br>
Motivation: all these adversarial attack methods involve perturbing input features, so defending against one type of perturbation may result in better defense against other perturbations as well<br>

Evaluation criteria: a model trained against a single/multiple type(s) of attack evaluated on examples crafted by another attack<br>

<b>Overview of Experiments</b><br><br>

Rerun some experiments from last time with full MNIST dataset

Combinations of FGSM, Virtual Adversarial Training, and Basic Iterative method trained model tested on samples crafted by both types

DeepFool Adversarial Training

Combined Enhanced JSMA and ElasticNet

Combined model with adversarial training on all 6 attack types


<b>Conclusions</b><br><br>

Adversarial training never decreases model test accuracy on legitimate examples

Training on examples from one type of attack can aid against another, but the effectiveness varies

JSMA attacks tend to have a high success rate even with adversarial training (This might be JSMA attack is very intricate and sensitive to input data)

Also observed this for DeepFool and ElasticNet
