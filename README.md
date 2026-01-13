
**Design choices:**
- ReLU activations
- Batch Normalization for stability
- Dropout for overfitting control
- Residual skip connection for improved gradient flow

**Output layer:**
- 2 neurons (DR / No DR)
- Softmax activation
- Loss: **Categorical Cross-Entropy**

---

### 7️⃣ Model Compilation

- **Optimizer:** Adamax  
- **Learning Rate:** 0.0005  

**Metrics tracked during training:**
- Accuracy
- Precision
- Recall

---

### 8️⃣ Automated Training Pipeline

The training process is fully automated and supports **multi-backbone experimentation**.

Key features:
- Trains multiple CNN backbones in a loop
- Automatic checkpointing (`latest_model.keras`)
- Resume training from the last saved epoch
- Epoch-wise logging of training and validation metrics to CSV files

---

### 9️⃣ Advanced Metrics Logging (Custom Callback)

At the end of every epoch, predictions are generated on the **validation set** to compute clinically meaningful metrics:

- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Sensitivity
- Specificity
- False Positive Rate (FPR)
- False Negative Rate (FNR)

Each log entry includes:
- Timestamp
- Epoch number
- Learning rate

Metrics are saved to:
- Model-specific CSV files
- A master CSV aggregating all backbones

---

### 🔟 Visualization & Model Comparison

The pipeline automatically generates:
- Training vs Validation accuracy curves
- Training vs Validation loss curves
- Automatic highlighting of the best epoch

After all models are trained:
- Cross-model comparison plots are generated to identify the best-performing backbone

---

### 1️⃣1️⃣ Final Model Evaluation

The **test set is used exactly once** to prevent data leakage.

Computed metrics:
- Accuracy
- Precision
- Recall
- F1-score

Additional outputs:
- Confusion Matrix
- Clinical safety metrics (Sensitivity, Specificity, FPR, FNR)

---

### 1️⃣2️⃣ Model Saving Logic

Models are saved **only if test accuracy exceeds 90%**, ensuring deployment-grade quality.

Saved format:
- `.keras` (production-ready and deployment-friendly)

---

## 📊 Key Results

The best-performing models achieve:
- **Accuracy:** up to 99%+
- **High Sensitivity:** minimizing missed DR cases
- **High Specificity:** reducing unnecessary referrals
- **Stable generalization across validation and test sets**

The design prioritizes **clinical safety over superficial accuracy gains**.

---

## ⚡ Deployment Readiness

- CPU-efficient inference
- Portable `.keras` format
- Suitable for edge and point-of-care screening systems
- On-device inference ensures **patient data privacy**

---

## 🩺 Clinical Significance

This system is designed for **screening**, not diagnosis. By maintaining very low false negative rates and reliable generalization, it supports early referral and triage in real-world healthcare settings.

The modular design allows:
- Easy retraining
- Backbone replacement
- Adaptation to new datasets or imaging devices

---

## 🔮 Future Work

- Multi-class DR severity grading
- Lesion-level explainability (Grad-CAM)
- Cross-dataset validation (EyePACS, Messidor)
- Edge deployment using TensorFlow Lite

---

## 👩‍💻 Authors

**Manushi Bombaywala**  
Palak Jethwani  
Arumuga Arun R  

School of Computer Science and Engineering  
Vellore Institute of Technology, India  

---

## ⭐ Acknowledgements

This project uses the **APTOS-2019 Blindness Detection dataset** made publicly available via Kaggle.  
Inspired by advances in edge AI, medical imaging, and efficient deep learning for healthcare.

If you find this work useful, please consider starring the repository.
