# 🩺 Lightweight Diabetic Retinopathy Detection for Point-of-Care Devices

Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness worldwide. Early detection through retinal screening can significantly reduce vision loss, yet manual grading of fundus images is time-consuming, subjective, and difficult to scale—especially in rural and resource-limited healthcare environments.

This project presents a **clinically reliable, lightweight diabetic retinopathy classification system** built using **ImageNet transfer learning**, a **custom residual classifier head**, **minimal but safe preprocessing**, and **automated multi-model training and evaluation**. The pipeline is designed with a strong focus on **real-world deployability, reproducibility, and diagnostic safety**.

---

## 🔍 Motivation

Most deep learning models for medical imaging focus primarily on accuracy, often resulting in large, compute-heavy architectures that are impractical for point-of-care deployment. In contrast, this work prioritizes:

- Low memory footprint  
- CPU-only execution  
- Real-time inference  
- Very low false negative rates (clinical safety)  

The objective is to bridge the gap between **research-grade performance** and **real-world clinical usability**.

---

## 🔹 End-to-End Diabetic Retinopathy Classification Pipeline

This section describes the complete pipeline used to build, train, evaluate, and select reliable DR classification models.

---

### 1️⃣ Dataset Loading & Labeling

Retinal fundus images are organized in **class-wise directories** (`DR` and `No_DR`).

The pipeline:
- Traverses directories using `os.listdir`
- Collects absolute image paths
- Assigns binary labels:
  - **DR → 1**
  - **No DR → 0**

All image paths and labels are stored in a **Pandas DataFrame**, ensuring traceability and seamless integration with data generators.

---

### 2️⃣ Dataset Splitting Strategy

The dataset is split using **stratified sampling** to preserve class balance:

- **70% Training**
- **15% Validation**
- **15% Test**

Key constraints:
- Validation and test sets are **never augmented**
- No shuffling during evaluation
- Ensures unbiased and clinically reliable performance estimation

---

### 3️⃣ Image Resolution & Input Shape

All images are resized to:

**256 × 256 × 3 (RGB)**

This ensures:
- Uniform input size
- Compatibility with pretrained ImageNet backbones
- Efficient balance between visual detail and computational cost

---

### 4️⃣ Image Preprocessing (On-the-Fly)

Preprocessing is performed dynamically using `ImageDataGenerator`.

**Applied to all images:**
- Pixel rescaling:  
  `pixel = pixel / 255` → normalized to `[0, 1]`

**Applied only to training images:**
- Brightness augmentation:  
  `brightness_range = (0.8, 1.2)`

**Explicitly avoided:**
- No rotations, flips, zooming, or cropping  
- No ImageNet mean–std normalization  
- No contrast enhancement (CLAHE)

📌 **Rationale:** Preserve clinical authenticity and avoid artificial distortions that could bias medical interpretation.

---

### 5️⃣ ImageNet Transfer Learning (Core Design)

Pretrained **ImageNet CNN backbones** (ResNet, Inception, DenseNet, etc.) are used as **feature extractors**.

- The original 1000-class ImageNet classification head is removed
- Only convolutional layers are retained

**Why ImageNet?**
ImageNet models learn transferable low- and mid-level features such as:
- Edges
- Textures
- Shapes  

These representations generalize effectively to retinal imaging tasks.

---

### 6️⃣ Custom Classification Head (Residual Design)

The ImageNet head is replaced with a **custom residual classifier head** optimized for DR detection.

**Architecture:**
512 → 256 → (Residual 256) → 128 → 64 → Output


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
