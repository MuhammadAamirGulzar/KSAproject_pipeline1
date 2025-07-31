# Adapting Foundation Models in Computational Pathology: MSI/MSS Status Prediction in Colorectal Cancer

## Overview

This repository contains the implementation of a comprehensive computational pathology pipeline for adapting foundation models to analyze Whole Slide Images (WSI) of colorectal cancer tissues. The project focuses on the extraction of visual embeddings (features) from histopathological image patches and the subsequent classification of microsatellite instability (MSI) and microsatellite stability (MSS) status, which are important biomarkers in colorectal cancer.

The research compares the performance of several state-of-the-art computer vision and foundation models for histopathology feature extraction and evaluates different aggregation methods and classification algorithms.

## Datasets

The research utilized two main datasets:
- **TCGA** (The Cancer Genome Atlas) - Colorectal cancer whole slide images
- **PAIP** (Pathology AI Platform) - Colorectal cancer whole slide images

## Methodology

### 1. Data Preprocessing
- WSI (.svs files) reading and processing using OpenSlide library
- Dynamic downsampling and conversion to .png format
- Patch extraction from tissue regions

### 2. Feature Extraction
Four models were used for feature extraction:
- **CONCH** ([MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH)) - A computational pathology foundation model
- **UNI** ([MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)) - A unified representation learning model for computational pathology
- **Virchow2** ([paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2)) - A pathology-specific vision model
- **Baseline** (ResNet34) - A standard computer vision backbone

### 3. Feature Aggregation
Two primary methods were implemented to convert patch-level features to slide-level representations:
- **Averaging Aggregation**: Simple Mean averaging of all patch features from a slide
- **Clustering Aggregation**: 
  - Clustering patches into 2 or 3 clusters
  - Averaging features within each cluster
  - Concatenating the cluster-level features to form the slide-level representation

### 4. Adaptation by Classification
Four different classifiers were trained and evaluated:
- **ANN** (Artificial Neural Network)
- **Linear** (Logistic Regression)
- **KNN** (K-Nearest Neighbors)
- **ProtoNet** (Prototypical Networks)

## Repository Structure

```
lab-calm-colonmsi-fm/
├── data_preprocessing/           # Scripts for WSI processing and patch extraction
├── env files/                    # Environment configuration
├── Feature_Extraction_Using_FM/  # Feature extraction using foundation models
│   ├── CONCH-main/               # CONCH model implementation
│   ├── UNI_main/                 # UNI model implementation
│   └── TissueClassifier_CRC100K/ # Tissue classification
└── WSI_Classification/           # Slide-level classification
    ├── Averaging/                # Models and results for averaging aggregation
    └── Clustering/               # Models and results for clustering aggregation
```

## Installation and Setup

1. Clone this repository
```bash
git clone https://github.com/lab-calm/ColonMSI-FM
cd lab-calm-colonmsi-fm
```

2. Install the required dependencies
```bash
conda env create -f env\ files/tcga_env.yml
conda activate tcga_env
```

## Usage

### Data Preprocessing
```bash
cd data_preprocessing
jupyter notebook data_downsample.ipynb
```

### Feature Extraction
```bash
cd Feature_Extraction_Using_FM
jupyter notebook CONCH_WSI_Avg_Feature_Extraction.ipynb
```

### Classification
```bash
cd WSI_Classification
jupyter notebook TCGA-CV_WSI_Classification_Averaging.ipynb
```

## Results

The repository includes comprehensive evaluation metrics for all experiments, including:
- Performance comparisons between different feature extraction models
- Evaluation of different aggregation methods
- Assessment of various classification algorithms
- Cross-validation results for the TCGA dataset
- External validation results using the PAIP dataset

## Acknowledgements

This research builds upon several open-source models and libraries:

### Foundation Models
- **CONCH**: MahmoodLab/CONCH (https://huggingface.co/MahmoodLab/CONCH)

- **UNI**: MahmoodLab/UNI (https://huggingface.co/MahmoodLab/UNI)

- **Virchow2**: paige-ai/Virchow2 (https://huggingface.co/paige-ai/Virchow2)

### Libraries
- OpenSlide: For handling whole slide images
- PyTorch: Deep learning framework
- scikit-learn: For machine learning algorithms
- NumPy/Pandas: For data manipulation and analysis

## Contact

For any questions or inquiries, please open an issue on this repository.

## Citation
- To be added soon

