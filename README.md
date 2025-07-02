# ECG Denoising with Convolutional Denoising Autoencoder and Batch Attention Mechanism (CDAE-BAM)

## Project Overview
This project implements a **Convolutional Denoising Autoencoder with Batch Attention Mechanism (CDAE-BAM)** to remove common noise types (EMG, baseline wander, and powerline interference) from ECG signals using the **MIT-BIH Arrhythmia Database**. The goal is to enhance ECG signal quality for downstream clinical applications, such as arrhythmia detection, by leveraging advanced deep learning techniques. This work demonstrates expertise in Biomedical Signal Processing and deep learning, aligning with cutting-edge research in ECG analysis.

## Motivation
ECG signals are often corrupted by noise, which can obscure critical diagnostic features. Effective denoising is essential for accurate clinical interpretation. This project introduces a novel CDAE-BAM model that combines convolutional layers for feature extraction with a batch attention mechanism to focus on relevant signal patterns, achieving superior denoising performance.

## Dataset
The project uses the **MIT-BIH Arrhythmia Database**, a widely-used benchmark dataset available from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/). The dataset includes 48 half-hour ECG recordings with two leads (MLII and V5). For this project, we use the MLII lead from a subset of records (e.g., records 100–104) to demonstrate the denoising pipeline.

## Methodology
The project follows an end-to-end pipeline for ECG denoising:

1. **Data Acquisition**: Download ECG records from the MIT-BIH Arrhythmia Database using the `wfdb` Python library.
2. **Preprocessing**: Normalize ECG signals using StandardScaler and segment them into fixed-length windows (256 samples) for model training.
3. **Noise Simulation**: Add synthetic noise (EMG, baseline wander, and powerline interference) to clean ECG signals to create noisy inputs for training.
4. **Model Architecture**:
   - **Convolutional Denoising Autoencoder (CDAE)**: Uses convolutional layers for encoding and decoding to capture spatial patterns in ECG signals.
   - **Batch Attention Mechanism (BAM)**: Applies global average pooling and dense layers to compute attention weights, enhancing the model's focus on relevant features.
   - The model is trained to reconstruct clean ECG signals from noisy inputs using Mean Squared Error (MSE) loss.
5. **Training**: Train the model on segmented ECG windows with Adam optimizer for 50 epochs.
6. **Evaluation**: Assess denoising performance using Signal-to-Noise Ratio (SNR) and visualize clean, noisy, and denoised signals.
7. **Visualization**: Plot original, noisy, and denoised ECG signals to demonstrate model effectiveness.

## Prerequisites
To run this project, ensure you have the following dependencies installed:
- Python 3.8+
- Libraries: `numpy`, `pandas`, `wfdb`, `tensorflow`, `scikit-learn`, `matplotlib`

Install dependencies using:
```bash
pip install numpy pandas wfdb tensorflow scikit-learn matplotlib
```

## Results
- **Quantitative Evaluation**: The model significantly improves SNR:
  - SNR (Noisy vs Clean): 6.44 dB
  - SNR (Denoised vs Clean): 10.84 dB
  - This represents a ~4.4 dB improvement, demonstrating effective noise removal.
- **Qualitative Evaluation**: Visualizations show effective removal of noise while preserving ECG morphology (e.g., QRS complexes).

### Sample Visualization
![image](https://github.com/user-attachments/assets/14b9bf94-48b1-4420-8861-d49b23a13f5d)

*Figure: Comparison of clean, noisy, and denoised ECG signals.*



