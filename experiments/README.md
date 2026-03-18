# Experiments

This folder contains standalone scripts that document experiments conducted during the
project's development phase. They are **not** part of the production system but provide
evidence of iterative research and design decisions.

## Contents

| Script | Purpose | Key Finding |
|--------|---------|-------------|
| `ocr_comparison.py` | Compare Tesseract vs. EasyOCR on handwriting samples | OCR accuracy on children's handwriting was too low (~30-50%) to use raw text as a reliable feature |
| `svm_tuning.py` | Hyperparameter tuning for SVM classifier | RBF kernel with C=10, gamma=0.01 performed best; motivates including SVM in the model registry |
| `text_classification.py` | Classify extracted text for reading-error patterns | Text-level features were noisy due to OCR errors; led to pivoting toward image-level CV features |

## Running

```bash
# From the project root
cd experiments
python ocr_comparison.py --image-dir ../samples/handwriting
python svm_tuning.py --samples 2000
python text_classification.py --csv ../data/extracted_text.csv
```

## Takeaways

1. **OCR is not reliable enough** for children's handwriting to extract text features
   directly. The project pivoted to using OpenCV-based image features instead.
2. **SVM with RBF** is competitive with ensemble methods on this feature set, justifying
   its inclusion in the model registry alongside Random Forest and Gradient Boosting.
3. **Text classification** on OCR output yielded high variance; the multimodal
   image + speech approach proved more robust.
