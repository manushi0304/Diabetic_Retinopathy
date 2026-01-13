🩺 Lightweight Diabetic Retinopathy Detection for Point-of-Care Devices

Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness, particularly affecting populations with limited access to specialized ophthalmology services. Early screening can significantly reduce the risk of vision loss, but manual grading of retinal fundus images is labor-intensive, subjective, and difficult to scale.

This project presents a lightweight, real-time diabetic retinopathy screening system designed specifically for point-of-care and edge deployment. The system combines deep convolutional neural networks (CNNs) for powerful feature extraction with post-training quantization and classical machine learning classifiers to achieve high accuracy while remaining computationally efficient.

🔍 Motivation and Problem Statement

Most deep learning models for medical imaging prioritize accuracy but overlook deployability constraints such as memory footprint, inference latency, and CPU-only execution. In real-world screening environments—such as rural clinics, mobile health vans, and primary-care centers—these constraints are critical.

The goal of this work is to bridge the gap between high-performing DR classifiers and practical clinical deployment by:

Reducing model size without compromising diagnostic safety

Maintaining very high sensitivity to avoid missed disease cases

Ensuring fast and reliable inference on low-resource hardware

🧠 Methodology Overview

The proposed system follows a hybrid inference approach. Instead of relying on a large end-to-end deep network, the pipeline separates representation learning from classification.

First, a deep CNN pretrained on ImageNet is fine-tuned on retinal fundus images to learn rich, hierarchical visual features relevant to diabetic retinopathy, such as microaneurysms, hemorrhages, and exudates. The final classification layers of the CNN are then removed, and the network is repurposed as a feature extractor.

To make the system suitable for deployment, the trained CNN models are converted to TensorFlow Lite format and optimized using post-training quantization. Multiple quantization strategies (FP32, FP16, and INT8) are explored, with FP16 offering the best balance between accuracy retention and model compression.

Once features are extracted from the quantized CNN, they are standardized and optionally reduced in dimensionality using Principal Component Analysis (PCA). These compact feature vectors are then classified using lightweight machine learning models such as Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Random Forests (RF).

This hybrid design ensures that most of the computational burden lies in a compact, optimized CNN, while the final decision-making remains fast, interpretable, and memory-efficient.

🧪 Models and Experimental Design

The system evaluates multiple state-of-the-art CNN backbones commonly used in medical image analysis, including:

DenseNet121

EfficientNetV2B0

MobileNetV2

InceptionV3

ResNet50V2

NASNetMobile

All models are trained and evaluated on the APTOS-2019 Blindness Detection dataset, which consists of retinal fundus images collected from real-world screening programs in India. To reflect realistic screening conditions, the task is framed as a binary classification problem:
No Diabetic Retinopathy vs Diabetic Retinopathy.

Careful preprocessing is applied to preserve clinically relevant details while minimizing artificial enhancement. Duplicate images are removed, patient-level splits are maintained, and multiple clinically meaningful metrics are reported.

📊 Results and Key Findings

Across all evaluated backbones, the models achieve strong classification performance, with accuracy ranging between 97.23% and 99.44%. However, accuracy alone is not sufficient for clinical screening. Therefore, the system emphasizes:

Sensitivity (Recall for DR) to minimize missed disease

Specificity to avoid unnecessary referrals

False Positive Rate (FPR) and False Negative Rate (FNR) for safety analysis

Among all configurations, the DenseNet121 FP16 feature extractor combined with an SVM classifier emerged as the best-performing hybrid model. This configuration achieved:

99.25% Accuracy

99.22% F1-Score

99.35% Sensitivity

99.09% Specificity

Crucially, this performance was achieved with a model size of just 14.77 MB, down from the original 90.4 MB DenseNet121 model, demonstrating that substantial compression is possible without sacrificing diagnostic reliability.

⚡ Deployment and Efficiency

The optimized models are designed to run efficiently on CPU-only environments, making them suitable for real-time screening. Average inference times range from approximately 75 ms to 170 ms per image, depending on the backbone and classifier used.

By exporting the models in TensorFlow Lite format, the system is portable and easily integrable into mobile applications, embedded systems, or clinical screening devices. All processing is performed on-device, ensuring data privacy and eliminating the need for cloud-based inference.

🩺 Clinical Relevance

From a screening perspective, the proposed system prioritizes patient safety. Very low false negative rates ensure that individuals with diabetic retinopathy are rarely missed, while high specificity limits unnecessary referrals and reduces clinician workload.

The modular nature of the pipeline allows the classifier head to be easily swapped or retrained without modifying the underlying feature extractor, making the system adaptable to new datasets, imaging devices, or clinical requirements.

🔮 Future Directions

While the current work focuses on binary DR screening, future extensions may include:

Multi-class grading of DR severity

Lesion-level explainability using Grad-CAM

Cross-dataset generalization studies

Deployment on mobile and embedded platforms

👩‍💻 Authors

Manushi Bombaywala
Palak Jethwani
Arumuga Arun R

School of Computer Science and Engineering
Vellore Institute of Technology, India

⭐ Acknowledgements

This project uses the APTOS-2019 Blindness Detection dataset made publicly available via Kaggle. The work is inspired by recent advances in edge AI, medical imaging, and efficient deep learning.
