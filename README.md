# 🩺 Lightweight Diabetic Retinopathy Detection for Point-of-Care Devices

Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness worldwide. Early screening and timely referral can significantly reduce vision loss, yet manual grading of retinal fundus images is slow, subjective, and difficult to scale—especially in rural and resource-constrained healthcare settings.

This project presents a **lightweight, real-time diabetic retinopathy screening system** optimized for **edge and point-of-care deployment**. The system combines **deep convolutional neural networks (CNNs)** with **post-training quantization and classical machine learning classifiers**, achieving high diagnostic accuracy while maintaining a small memory footprint and fast inference.

---

## 🔍 Motivation

Most deep learning-based medical imaging systems focus primarily on maximizing accuracy, often resulting in large, compute-heavy models that are impractical for real-world clinical deployment. In contrast, this work emphasizes **deployability**, addressing key constraints such as:

- Limited memory and compute availability
- CPU-only execution environments
- Real-time inference requirements
- High clinical safety (low false negatives)

The objective is to bridge the gap between research-grade performance and real-world usability.

---

## 🧠 Methodology

The proposed system follows a **hybrid inference pipeline** that decouples feature extraction from classification.

Pretrained CNN backbones are fine-tuned on retinal fundus images to learn robust, hierarchical representations of diabetic retinopathy indicators such as microaneurysms, hemorrhages, and exudates. After training, the classification layers are removed, and the CNN is used solely as a **feature extractor**.

To enable deployment on resource-constrained devices, the trained CNNs are converted into **TensorFlow Lite** format and optimized using **post-training quantization**. Multiple quantization strategies (FP32, FP16, INT8) are explored, with FP16 providing the best balance between compression and performance stability.

The extracted feature vectors are standardized and optionally reduced in dimensionality using **Principal Component Analysis (PCA)**. Lightweight machine learning classifiers—**Support Vector Machines (SVM)**, **K-Nearest Neighbors (KNN)**, and **Random Forests (RF)**—are then trained on these features to perform the final classification.

This design keeps the deep learning component compact and efficient, while ensuring fast, interpretable decision-making.

---

## 🧪 Models and Dataset

### CNN Backbones Evaluated
- DenseNet121  
- EfficientNetV2B0  
- MobileNetV2  
- InceptionV3  
- ResNet50V2  
- NASNetMobile  

### Dataset
The models are evaluated using the **APTOS-2019 Blindness Detection dataset**, consisting of retinal fundus images collected from large-scale screening programs in India.

The task is framed as a **binary classification problem**:
- **No Diabetic Retinopathy (No-DR)**
- **Diabetic Retinopathy (DR)**

The dataset is carefully cleaned by removing duplicates, maintaining patient-level splits, and applying minimal preprocessing to preserve clinically relevant features.

---

## 📊 Results and Performance

All evaluated CNN backbones demonstrate strong performance, achieving classification accuracy between **97.23% and 99.44%**.

The best-performing configuration is a **DenseNet121 FP16 feature extractor combined with an SVM classifier**, which achieves:

- **Accuracy:** 99.25%  
- **F1-Score:** 99.22%  
- **Sensitivity:** 99.35%  
- **Specificity:** 99.09%  
- **Model Size:** 14.77 MB  

This represents a **~6× reduction in model size** compared to the original DenseNet121 model (90.4 MB), with no loss in clinical reliability.

---

## ⚡ Efficiency and Deployment

The optimized models run efficiently on **CPU-only environments**, with average inference times ranging from approximately **75 ms to 170 ms per image**, depending on the backbone and classifier.

By exporting the models in **TensorFlow Lite format**, the system is portable and suitable for integration into:
- Point-of-care diagnostic devices
- Mobile screening applications
- Embedded healthcare systems

All inference is performed **on-device**, ensuring patient data privacy and eliminating dependency on cloud infrastructure.

---

## 🩺 Clinical Significance

From a clinical perspective, the system prioritizes **high sensitivity** to minimize missed DR cases, while maintaining high specificity to avoid unnecessary referrals. Reporting metrics such as **False Positive Rate (FPR)** and **False Negative Rate (FNR)** ensures transparency and reliability for screening applications.

The modular design allows easy replacement or retraining of the classifier head without modifying the feature extractor, making the system adaptable to new datasets or imaging devices.

---

## 🔮 Future Work

Potential extensions of this work include:
- Multi-class grading of diabetic retinopathy severity
- Lesion-level explainability using Grad-CAM
- Cross-dataset validation on EyePACS and Messidor
- Deployment on mobile and embedded hardware platforms

---

## 👩‍💻 Authors

**Manushi Bombaywala**  
Palak Jethwani  
Arumuga Arun R  

School of Computer Science and Engineering  
Vellore Institute of Technology, India  

---

## ⭐ Acknowledgements

This project uses the **APTOS-2019 Blindness Detection dataset** provided via Kaggle. The work is inspired by recent advances in edge AI, medical image analysis, and efficient deep learning for healthcare.

If you find this project useful, please consider starring the repository.
