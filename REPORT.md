# Report

## Results Summary

| Run | Acc | Macro-F1 | Weighted-F1 | Top-5 |
|---|---:|---:|---:|---:|
| effb0_224_adam_cosine_ls05 | 0.924 | 0.916 | 0.922 | 0.995 |
| vitb16_224_freeze | 0.883 | 0.883 | 0.876 | 0.982 |
| hog_svm_rbf | 0.137 | 0.017 | 0.081 | - |
| resnet18_224_sgd_cosine_ls01 | 0.901 | 0.889 | 0.900 | 0.991 |

## Results (auto-generated)

**Overall results**
| Run | Acc | Macro-F1 | Weighted-F1 | Top-5 |
|---|---:|---:|---:|---:|
| vitb16_224_fullft | 0.935 | 0.931 | 0.934 | 0.995 |
| effb0_224_adam_cosine_ls05 | 0.924 | 0.916 | 0.922 | 0.995 |
| resnet18_noaug_bs256 | 0.921 | 0.894 | 0.921 | 0.988 |
| resnet18_224_sgd_cosine_ls01 | 0.901 | 0.889 | 0.900 | 0.991 |
| resnet18_128px | 0.867 | 0.839 | 0.867 | 0.972 |
| resnet18_aug_bs256 | 0.867 | 0.837 | 0.866 | 0.974 |
| vitb16_224_freeze | 0.848 | 0.822 | 0.835 | 0.975 |
| resnet18_64px | 0.760 | 0.688 | 0.757 | 0.932 |
| hog_svm_rbf | 0.137 | 0.017 | 0.081 | - |


### Ablation Size Resnet18

**Ablation: ResNet-18 input size**
| Run | Acc | Macro-F1 | Weighted-F1 | Top-5 |
|---|---:|---:|---:|---:|
| resnet18_128px | 0.867 | 0.839 | 0.867 | 0.972 |
| resnet18_224_sgd_cosine_ls01 | 0.901 | 0.889 | 0.900 | 0.991 |
| resnet18_64px | 0.760 | 0.688 | 0.757 | 0.932 |


### Ablation Aug Resnet18

**Ablation: ResNet-18 augmentation**
| Run | Acc | Macro-F1 | Weighted-F1 | Top-5 |
|---|---:|---:|---:|---:|
| resnet18_aug_bs256 | 0.867 | 0.837 | 0.866 | 0.974 |
| resnet18_noaug_bs256 | 0.921 | 0.894 | 0.921 | 0.988 |


### Ablation Vit Freeze Full

**Ablation: ViT freezing vs full FT**
| Run | Acc | Macro-F1 | Weighted-F1 | Top-5 |
|---|---:|---:|---:|---:|
| vitb16_224_freeze | 0.848 | 0.822 | 0.835 | 0.975 |
| vitb16_224_fullft | 0.935 | 0.931 | 0.934 | 0.995 |

