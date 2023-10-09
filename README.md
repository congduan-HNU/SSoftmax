# Enhancing Cross-Dataset Performance of Distracted Driving Detection With Score-Softmax Classifier

### [paper](https://arxiv.org/abs/2209.09178)


## Abstract

Deep neural networks enable real-time monitoring of in-vehicle driver, facilitating the timely prediction of distractions, fatigue, and potential hazards. This technology is now integral to intelligent transportation systems. Recent research has exposed unreliable cross-dataset end-to-end driver behavior recognition due to overfitting, often referred to as ``shortcut learning", resulting from limited data samples. In this paper, we introduce the Score-Softmax classifier, which addresses this issue by enhancing inter-class independence and Intra-class uncertainty. Motivated by human rating patterns, we designed a two-dimensional supervisory matrix based on marginal Gaussian distributions to train the classifier. Gaussian distributions help amplify intra-class uncertainty while ensuring the Score-Softmax classifier learns accurate knowledge. Furthermore, leveraging the summation of independent Gaussian distributed random variables, we introduced a multi-channel information fusion method. This strategy effectively resolves the multi-information fusion challenge for the Score-Softmax classifier. Concurrently, we substantiate the necessity of transfer learning and multi-dataset combination. We conducted cross-dataset experiments using the SFD, AUCDD-V1, and 100-Driver datasets, demonstrating that Score-Softmax improves cross-dataset performance without modifying the model architecture. This provides a new approach for enhancing neural network generalization. Additionally, our information fusion approach outperforms traditional methods.

### Data Preparation



## Citation

If you find S-Softmax Classifier beneficial or relevant to your research, please kindly recognize our efforts by citing our paper:

```bibtex
@article{Ma2022MultiTaskVT,
  title={Multi-Task Vision Transformer for Semi-Supervised Driver Distraction Detection},
  author={Yunsheng Ma and Ziran Wang},
  journal={arXiv},
  year={2022}
}
```
