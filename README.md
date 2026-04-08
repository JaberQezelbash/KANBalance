# KANBalance
## Kolmogorov-Arnold Network Mitigates Class Imbalance

<img width="800" alt="kanbalance_figure" src="https://github.com/JaberQezelbash/KANBalance/blob/main/assets/KANBalance.svg">

This repository contains the KANBalance framework, as presented in the paper:

> **KANBalance: Kolmogorov-Arnold Network Mitigates Class Imbalance**  
> *by Qezelbash-Chamak et al.*  
> [[Paper Link]](https://doi.org/10.1016/j.patcog.2025.112325)

---


## Motivation
Class imbalance is a common but critical challenge in machine learning, particularly in medical image analysis where minority-class samples (e.g., rare diseases or normal scans) are severely underrepresented. Standard techniques either oversample, undersample, or modify the loss to counteract skewed distributions. However, these approaches often fail to fundamentally enhance how models represent the subtle features of minority classes. KANBalance tackles this by combining (1) [Kolmogorov–Arnold Networks (KANs)](https://arxiv.org/abs/2404.19756), which replace fixed activations with univariate spline expansions on each edge—allowing the model to zoom in on narrow or complex feature intervals that characterize minority samples; (2) [Focal Loss](https://arxiv.org/abs/1708.02002), which amplifies the training signal for harder-to-classify (often minority) examples. By synergizing these two ideas, KANBalance reshapes feature representations to focus on minority classes, achieving robust performance gains over conventional resampling or cost-sensitive baselines. 


## Table of Contents
1. [Introduction](#introduction)  
2. [Methodology Overview](#methodology-overview)  
3. [Experiments](#experiments)  
4. [Implementation & Codes](#implementation--codes)  
5. [Installation & Requirements](#installation--requirements)  
6. [Configurations](#configurations)  
7. [Citation](#citation)  
8. [Contact](#contact)  
9. [Author's Note](#authors-note)



## Introduction
Addressing class imbalance is vital in tasks like medical image classification, where one label (e.g., “normal”) might represent only a fraction of all available scans. Traditional resampling methods (SMOTE, ADASYN, etc.) can introduce overlaps or discard valuable data, whereas purely loss-driven fixes (e.g., cost-sensitive or focal-based) may struggle to adapt the underlying feature representation itself.  

**Kolmogorov–Arnold Networks (KANs)** build on the theorem that any multivariate continuous function can be expressed as sums of univariate functions. Instead of using a shared activation function across neurons, KAN places a **unique spline-based function** on each edge, allowing the network to learn localized transformations directly from data. When combined with **Focal Loss**, the network’s capacity to focus on underrepresented and difficult examples is amplified—leading to an adaptive, minority-focused modeling.  

KANBalance, therefore, aims to:
- **Reinforce** the learning signal on challenging minority instances, thanks to the focusing parameter in Focal Loss.
- **Refine** local decision boundaries via univariate splines, ensuring subtle minority features are captured and not overshadowed by the majority class.
- **Stabilize** training via mild regularization and early stopping—preventing overfitting to local segments.


## Methodology Overview

### 1. Univariate Splines (KAN)
Traditional CNN or MLP architectures rely on the same activation function across all edges. KAN instead assigns a **B-spline expansion** to each edge, with learnable coefficients and partially learnable knot offsets:
- **Localized Control**: Spline segments affect only local neighborhoods, enabling the model to capture narrow minority-class patterns.
- **Adaptable Knots**: Shifting knot positions via backpropagation allows the network to "zoom in" on critical feature intervals.

### 2. Focal Loss for Imbalance
Rather than the standard cross-entropy, KANBalance adopts **Focal Loss**:
- **Amplified Gradients**: Misclassified minority samples gain higher gradient importance (controlled by parameter \(\gamma\)).
- **Balanced Class Weights**: Minority vs. majority misclassifications can be weighted via \(\alpha\), reducing skew.

### 3. End-to-End Training
KANBalance is trained end-to-end on a CPU or GPU. The synergy arises when Focal Loss pushes the model to update spline parameters specifically where minority examples are confused, resulting in finer discriminations between classes. Implementation details include:
- **Early Stopping**: to avoid overfitting the spline expansions.
- **Regularization**: mild weight decay on both linear weights and spline coefficients to control complexity.
- **Stable Learning Rate**: small step size (\(\sim 1\times 10^{-4}\)) ensures gradient spikes from Focal Loss do not destabilize the spline basis.



## Experiments

We benchmark KANBalance on two heavily imbalanced medical imaging datasets:

1. **Chest X-Ray (Pneumonia) Dataset**  
   - Task: Detect pneumonia vs. normal lungs.  
   - Imbalance: 4,273 pneumonia images vs. 1,583 normal.  
   - Results:  
     - Accuracy: ~96.67%  
     - F1 Score: ~95.57%  
     - AUC: ~97.37%  
   - Outperforms SMOTE, ADASYN, Dual Focal Loss, Class-Balanced Loss, and random oversampling/undersampling methods.

2. **Brain MRI (Tumor) Dataset**  
   - Task: Binary classification: Tumor (aggregated) vs. No Tumor.  
   - Imbalance: 2,475 tumor images vs. 822 no-tumor.  
   - Results:  
     - Accuracy: ~96.17%  
     - F1 Score: ~95.37%  
     - AUC: ~96.87%  
   - Consistent improvements over both data-level (SMOTE, ADASYN) and algorithm-level (Focal variants) baselines.

**Ablation Studies** also confirm that:
- **Learnable Knots** provide stronger local adaptivity than fixed spline positions.
- **Focal Loss** drastically boosts minority recall vs. plain cross-entropy.
- **Combined Approach** (KAN + Focal) maximizes synergy, achieving the highest precision, recall, and specificity.



## Implementation & Codes
All relevant source files will be provided in this repository’s main branch:

- **`datasets/`**: Scripts for loading X-ray & MRI datasets, along with augmentations (random flips, mild rotations).
- **`models/`**: Implementation of KANBalance layers, univariate spline expansions, and the integration with standard CNN backbones.
- **`losses/`**: Focal Loss variants, plus standard cross-entropy for reference.
- **`train.py`**: Main training loop with early stopping, logging, and evaluation.
- **`ablation_studies/`**: Scripts to toggle between different KAN configurations (fixed vs. learnable knots) and different loss functions (cross-entropy vs. Focal).

Visit [KANBalance codes](https://github.com/JaberQezelbash/KANBalance) for detailed usage instructions.

Note: We primarily demonstrate CPU-based training for smaller to medium datasets, but GPU usage is recommended for larger-scale experiments.



## Installation & Requirements

An example environment setup:

```bash
python==3.9.0
torch==2.0.1
numpy==1.24.4
pandas==2.0.1
scikit_learn==1.1.3
matplotlib==3.6.2
tqdm==4.66.2
sympy==1.11.1
setuptools==65.5.0
pyyaml
# etc.
```

## Citation

If you use KANBalance in your work, please cite this paper as follows:

```bibtex
@article{qezelbash2025KANBalance,
  title={KANBalance: Kolmogorov-Arnold Network Mitigates Class Imbalance},
  author = {Jaber Qezelbash-Chamak and Karen Hicklin and Minhee Kim},
  journal={Pattern Recognition},
  volume = {171},
  pages = {112325},
  year = {2026},
  issn = {0031-3203},
  doi = {https://doi.org/10.1016/j.patcog.2025.112325}
}
```


## Contact
For any questions related to KANBalance, please contact:
[qezelbashc.jaber@ufl.edu](qezelbashc.jaber@ufl.edu)


## Author's Note
I appreciate your interest in KANBalance. 
This framework is motivated by the urgent need for more adaptive and minority-focused solutions in medical imaging and other highly imbalanced domains.
Feedback and collaboration inquiries are welcome!
