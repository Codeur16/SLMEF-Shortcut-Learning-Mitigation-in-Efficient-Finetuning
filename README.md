# Shortcut-Learning-Mitigation-in-Efficient-Finetuning
A Neuro-Symbolic Approach for Reducing Shortcut Learning in Large Language Models
## Presentation
Large Language Models (LLMs) have opened new perspectives in artificial intelligence, but their reliability is often undermined by the phenomenon of shortcut learning. This issue leads models to rely on superficial and easily exploitable cues in the data, at the expense of true and generalizable understanding. Mitigating this bias is a crucial challenge to ensure model robustness in varied contexts.
The SLMEF project focuses on experimenting with the tradeoff between robustness against shortcut learning and the computational efficiency of fine-tuning, especially in resource-limited environments. By combining efficient fine-tuning techniques with neuro-symbolic approaches integrating logical constraints, SLMEF aims to enhance model reliability while maintaining controlled resource usage. This approach is designed to enable more robust fine-tuning accessible to a wide range of models and users, even under hardware limitations.﻿
## Research Methodology
The research methodology of the SLMEF project involves combining parameter-efficient fine-tuning techniques with neuro-symbolic logical constraints integration, conducting systematic experiments to evaluate the tradeoff between shortcut learning mitigation and computational efficiency on large language models, using prepared datasets, adversarial benchmarks, and iterative rule tuning to optimize model robustness and performance, especially in resource-constrained environments.
## Objectives
The main objective of this work is to experimentally investigate the tradeoff between robustness against shortcut learning and computational efficiency during fine-tuning of large language models, by combining parameter-efficient fine-tuning with neuro-symbolic methods integrating logical constraints, to enable more reliable and resource-conscious adaptation of models in resource-limited environments.
## Approaches to test
Here are commonly tested approaches:
  * Benchmarking on standard datasets to verify baseline accuracy and general task performance.
  * Testing robustness on adversarial or out-of-distribution datasets designed to expose shortcut learning vulnerabilities.
  * Comparative evaluation against existing fine-tuning and shortcut mitigation methods, such as PEFT techniques alone or other neuro-symbolic approaches.
  * Ablation studies varying the presence of symbolic constraints or different hyperparameter settings to analyze their impact on robustness and efficiency.
  * Efficiency assessment including computational cost, memory usage, and training time to understand tradeoffs with robustness improvements.
## Experimentations
#### Experimentation environment
The experimentation environment for the project typically includes:
  * Cloud-based GPU virtual machines on Kaggle platform featuring 2 NVIDIA Tesla T4 GPUs (16GB each), dual-core Intel Xeon CPUs at 2.2 GHz, and 13GB RAM.
  * Local Ubuntu machine with Intel Core i5 CPU, 16GB RAM, and integrated Intel UHD Graphics GPU used primarily for data preprocessing.
  * Software stack includes Python, PyTorch 2.5, Hugging Face Transformers v4.53.3, and PEFT library for LoRA implementation.
  * Data used: MultiNLI for standard training; adversarial datasets HANS and MNLI-hard for robustness evaluation.
  * Evaluation metrics include model accuracy, robustness to shortcut learning, calibration, training time, and number of trainable parameters.
#### Experimental Architechture 
<img width="1418" height="746" alt="Frame 618 (1)" src="https://github.com/user-attachments/assets/fda794da-1345-419d-9342-f38303d1ef1c" />

## Contacts
  * charlesnjiosseu@gmail.com
  * author2@gmail.com
