# TCN vs CNN Model Comparison

This document provides a performance comparison between the Temporal Convolutional Network (TCN+Transformer) and the Convolutional Neural Network (CNN) used for the word-level Sign Language Detection task.

## Key Performance Metrics

### 1. Validation Set Only (Direct Generalization Test)
*Evaluating strictly on unseen validation data (2,566 samples)*

| Metric | TCN (Word Model V2) | CNN Model | Difference / Impact |
|--------|---------------------|-----------|---------------------|
| **Accuracy** | **85.54%** | 74.43% | TCN massively outperforms by **+11.11%** |
| **F1 Macro** | **85.29%** | 74.02% | TCN generalizes to complex unseen classes much better |

### 2. Full Dataset (Train + Val combo)
*Evaluating on all 12,829 samples*

| Metric | TCN (Word Model V2) | CNN Model | Difference / Impact |
|--------|---------------------|-----------|---------------------|
| **Accuracy** | **85.94%** | 84.22% | TCN consistently leads |
| **Word Error Rate (WER)** | 14.06% | 15.78% | TCN makes fewer errors |
| **Mean Jaccard** | 75.61% | 73.19% | TCN captures sequence overlap better |
| **Latency** | 1.452 ms | **1.222 ms** | CNN is faster (~0.2ms gap); both well under 30ms limit |
| **Parameters**| 388,771 | **363,042** | CNN is smaller by ~25K parameters |
| **FLOPs** | **15.0 M** | 20.67 M | TCN computes fewer FLOPs per pass |

## Visual Comparisons

Below are the graphical results comparing the models, generated from the standalone CNN training pipeline:

1. **Overall Metric Comparison**  
   ![Overall Comparison](c:\Users\raman\OneDrive\Desktop\Google SLD\comparison_results\overall_comparison.png)

2. **Per-Class F1 Score Comparison**  
   ![Per Class F1](c:\Users\raman\OneDrive\Desktop\Google SLD\comparison_results\per_class_f1.png)

3. **CNN Confusion Matrix**  
   ![CNN Confusion Matrix](c:\Users\raman\OneDrive\Desktop\Google SLD\comparison_results\cnn_confusion_matrix.png)

4. **CNN Training Curves**  
   ![Training Curves](c:\Users\raman\OneDrive\Desktop\Google SLD\comparison_results\training_curves.png)

## Conclusion
After addressing the attention pooling bottleneck in the Transformer architecture, the **TCN+Transformer (V2) Model** decisively outperforms the CNN model. When evaluated purely on unseen validation data, the TCN secures an **11.11% accuracy lead** (85.54% vs 74.43%) over the CNN. The attention pool successfully allows the TCN to focus on discriminative middle frames (solving issues like the "hat" vs "fireman" confusion). While the CNN retains a slight edge in raw parameter count (363K vs 388K) and inference speed (1.222ms vs 1.452ms), the TCN's massive generalization advantage and fewer arbitrary FLOPs make it undeniably the superior choice for deployment.

---

## List of Unnecessary Files
The following files/folders in the project directory are not required to run the core SignLens application (Flask server, models, UI) and could be removed if you are cleaning up the repository:

1. **`documentation.zip`** (File)
   - A compressed version of the `documentation` folder. The uncompressed folder already exists, making this ~3.2MB zip file redundant.
2. **`data_examine/`** (Directory)
   - Contains exploratory data analysis, similarity reports (`similarity_report.txt`, `analyze_all_signs.py`), and CSVs. These were helpful for dataset exploration (Step: "Analyzing Sign Similarities") but are not used in inference or training.
3. **`augment_alphabet_data.py`** (File)
   - A utility script used to artificially augment the alphabet dataset during the data preparation phase. Not needed for the final deployed application.
4. **`desktop.ini`** (File)
   - A hidden Windows configuration file that does not impact the Python project inside the root folder.
