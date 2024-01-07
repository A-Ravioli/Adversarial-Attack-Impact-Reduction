# Adversarial Attack Impact Reduction

## Overview

This repository contains code associated with the research paper titled ["Optimizing innate adversarial robustness on image recognition models in high risk scenarios."](https://www.academia.edu/85546902/Optimizing_innate_adversarial_robustness_on_image_recognition_models_in_high_risk_scenarios) The research explores the often-overlooked influence of optimization, activation, and architectural decisions on the adversarial robustness of convolutional neural networks (CNNs).

## Abstract

Convolutional Neural Networks have been and will continue to be utilized amongst a variety of high risk scenarios. To help protect these models, the majority of adversarial research has focused on improving the methods by which we adversarially train the existing convolutional models. This research neglects the effect the training of the original model may have on its adversarial robustness. In response, this work observes and defines how optimization, activation and architectural choices impact adversarial defense training and innate adversarial robustness. Additionally a framework for analyzing and scoring adversarial robustness as a function of classification accuracy is proposed. It is determined that regardless of the type of attack, Sigmoid activation, Adagrad, and Adamax optimizers weaken the model regardless of the method of attack. On the contrary, optimizers such as RMSProp are incredibly beneficial, outperforming the standard, Adam, on multiple adversarial attacks.

## Getting Started

To replicate the experiments or explore the analysis, simply download and run the notebooks.

## License

This project is licensed under the [MIT License](LICENSE), allowing for open collaboration and use.

## Acknowledgments

I appreciate the support and contributions from the research community. If you find this work valuable, please consider citing my paper (citation details can be found in the research paper). Feel free to open issues or pull requests for any improvements or corrections.
