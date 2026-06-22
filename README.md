**Thalamocortical and Autonomic Dysregulation in Cognitive Impairment: 1D-Inception Neural Network Analysis of Sleep Biosignals**

*Final Year Project 2026*

Overview
This repository contains the deployment codebase for a multi-modal deep learning pipeline designed to screen for Cognitive Impairment using raw sleep polysomnography (PSG) data. By fusing high-frequency electrophysiological waveforms (EEG and ECG) with static clinical demographics (Age, Sex, BMI), this project demonstrates the feasibility of automated, non-invasive cognitive diagnostics.

**Live Clinical Dashboard:** [Link to your Streamlit App Here]

Clinical Motivation
Standard sleep studies generate massive amounts of cortical and autonomic data. While traditionally used to diagnose sleep apnea or insomnia, micro-architectural anomalies in sleep stages are heavily correlated with neurodegenerative decline. This project bridges clinical physiology and machine learning by engineering an architecture capable of isolating these latent biomarkers.

Core Architecture: 1D-Inception Late Fusion
To handle the heterogeneous nature of the clinical data, a dual-pathway neural network was engineered natively in **PyTorch**:

1. **Time-Series Pathway:** A modified 1D-InceptionTime backbone processes the 100Hz resampled EEG and ECG continuous waveforms. The multi-scale convolutions capture both transient micro-arousals (sharp spikes) and long-wave baseline shifts.
2. **Tabular Pathway:** A Multi-Layer Perceptron (MLP) contextualizes the waveforms by processing the patient's Age, Sex, and Body Mass Index (BMI).
3. **Late Fusion:** The dense spatial features and clinical context vectors are concatenated and passed through a dropout-regularized classification head.

Model Training & Handling the Accuracy Paradox
The George B. Moody PhysioNet Challenge 2026 dataset presented a severe class imbalance (~93.4% Healthy vs. ~6.6% Impaired). 

To ensure clinical viability and prevent data leakage:
1. **Subject-Level Isolation:** A 3-Fold `GroupKFold` cross-validation strategy was utilized, grouping by `BDSPPatientID` to ensure the model was always evaluated on completely unseen patients.
2. **Class-Weighted Optimization:** A custom weighted Cross-Entropy Loss function was applied to heavily penalize the network for false negatives on the minority (Impaired) class.
3. **Final Performance:** The optimal fold achieved a **Validation AUROC of 0.918**, demonstrating robust discriminative power across independent clinical cohorts.

Technology Stack
1. **Deep Learning:** PyTorch, Torchvision
2. **Neurophysiology Data:** MNE-Python (Memory-mapped `.edf` streaming)
3. **Data Processing:** NumPy, Pandas, Scikit-Learn
4. **Web Deployment:** Streamlit Community Cloud
5. **Visualization:** Matplotlib, Seaborn

## Local Setup & Installation
To run this diagnostic dashboard on your local machine:

1. Clone the repository:
   ```bash
   git clone [https://github.com/PraiseAgboola/FINAL-YEAR-PROJECT-2026.git](https://github.com/PraiseAgboola/FINAL-YEAR-PROJECT-2026.git)
   cd FINAL-YEAR-PROJECT-2026
   ```bash
   git clone [https://github.com/PraiseAgboola/FINAL-YEAR-PROJECT-2026.git](https://github.com/PraiseAgboola/FINAL-YEAR-PROJECT-2026.git)
   cd FINAL-YEAR-PROJECT-2026
