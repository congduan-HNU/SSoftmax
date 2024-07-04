# Enhancing Cross-Dataset Performance of Distracted Driving Detection With Score Softmax Classifier and Dynamic Gaussian Smoothing Supervision

### [arxiv](https://arxiv.org/abs/2310.05202)
### [IEEE-TIV](https://ieeexplore.ieee.org/document/10553302)


## Abstract

Deep neural networks enable real-time monitoring of in-vehicle drivers, facilitating the timely prediction of distractions, fatigue, and potential hazards. This technology is now integral to intelligent transportation systems. Recent research has exposed unreliable cross-dataset driver behavior recognition due to a limited number of data samples and background noise. In this paper, we propose a Score-Softmax classifier, which reduces the model overconfidence by enhancing category independence. Imitating the human scoring process, we designed a two-dimensional dynamic supervisory matrix consisting of one-dimensional Gaussian-smoothed labels. The dynamic loss descent direction and Gaussian smoothing increase the uncertainty of training to prevent the model from falling into noise traps. Furthermore, we introduce a simple and convenient multi-channel information fusion method; it addresses the fusion issue among arbitrary Score-Softmax classification heads. We conducted cross-dataset experiments using the SFDDD, AUCDD, and the 100-Driver datasets, demonstrating that Score-Softmax improves cross-dataset performance without modifying the model architecture. The experiments indicate that the Score-Softmax classifier reduces the interference of background noise, enhancing the robustness of the model. It increases the cross-dataset accuracy by 21.34%, 11.89%, and 18.77% on the three datasets, respectively. The code is publicly available at https://github.com/congduan-HNU/SSoftmax.

### Data Preparation



## Citation

If you find S-Softmax Classifier beneficial or relevant to your research, please kindly recognize our efforts by citing our paper:

```bibtex
@misc{duan2023enhancing,
      title={Enhancing Cross-Dataset Performance of Distracted Driving Detection With Score-Softmax Classifier}, 
      author={Cong Duan and Zixuan Liu and Jiahao Xia and Minghai Zhang and Jiacai Liao and Libo Cao},
      year={2023},
      eprint={2310.05202},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@article{duanEnhancingCrossDatasetPerformance2024,
  title = {Enhancing {{Cross-Dataset Performance}} of {{Distracted Driving Detection With Score Softmax Classifier}} and {{Dynamic Gaussian Smoothing Supervision}}},
  author = {Duan, Cong and Liu, Zixuan and Xia, Jiahao and Zhang, Minghai and Liao, Jiacai and Cao, Libo},
  year = {2024},
  journal = {IEEE Trans. Intell. Veh.},
  pages = {1--14},
  issn = {2379-8904, 2379-8858},
  doi = {10.1109/TIV.2024.3412198},
}
```

