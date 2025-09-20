# Scene-text-recognition

The zip file   ```scene_text_file.zip```  contains the code file and the final report. The code file with name convention Batch_size{i} contains code and results with various batch sizes. The final code file consists of results for training and testing for the best model.
 * The synthetics dataset used for training: [https://www.robots.ox.ac.uk/~vgg/data/text/#sec-synth]
 * The dataset on which the model is tested: [https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset]
 * Extract the ```dataset.zip``` file which consists of around 12,797 synthetic images containing alphanumeric character. These images along with the labels are pre-processed before the training.

## Overview
Scene text recognition is a critical task in computer vision with applications in autonomous driving, augmented reality, digitizing documents, and accessibility for visually impaired individuals.  

This project implements a **Convolutional Recurrent Neural Network (CRNN)** for **scene text recognition**. The CRNN combines:
- **CNNs** for spatial feature extraction  
- **Bi-LSTMs** for sequential modeling  
- **CTC loss** for alignment-free training  

The model was trained on **synthetic datasets** and tested on the **IIIT5K benchmark dataset**, achieving robust performance across diverse real-world scenarios.
---

## Authors
- Umanshiva Ladva (ai22btech11016@iith.ac.in)  
- Rajiv Chaudhary (ai22btech11021@iith.ac.in)  
- Siddhesh Gholap (ai22btech11007@iith.ac.in)  
- Sudarshan Shivashankar (ai22btech11027@iith.ac.in)  
- Ruvva Suraj Kumar (ai22btech11022@iith.ac.in)  
*(IIT Hyderabad)*  

---

## Dataset

### 🔹 Training: Synthetic Dataset
- ~12,000 images  
- Includes varied fonts, sizes, and backgrounds  
- Generated for OCR tasks (subset of [Jaderberg et al.](https://arxiv.org/abs/1406.2227))  

### 🔹 Testing: IIIT5K Dataset
- 5,000 benchmark images  
- Covers diverse sources like **street signs, book pages, and scene text**  

### 🔹 Preprocessing Steps
- Grayscale conversion  
- Gaussian blurring (for denoising)  
- Resizing to **32 × 100**  
- Normalization (mean=0.5, std=0.5)  

---

## Model Architecture: CRNN

### 🔹 CNN Module
- Extracts spatial features  
- 5 convolutional blocks with ReLU, BatchNorm, and max-pooling  
- Output reshaped into sequential feature maps  

### 🔹 RNN Module
- **Bidirectional LSTM (Bi-LSTM)** with 2 layers  
- 256 hidden units per direction (512 total)  
- Captures forward and backward dependencies  

### 🔹 Transcription Layer
- **Connectionist Temporal Classification (CTC)**  
- Enables training without character-level alignment  

### 🔹 Decoding
- **Greedy decoding** to generate final predictions  

---

## Training Setup
- **Optimizer:** Adadelta (lr = 0.1)  
- **Loss:** CTC Loss  
- **Batch Sizes Tested:** 4, 8, 16, 32, 64  
- **Hardware:** NVIDIA GPU  

---

## 📊 Results

### Training Performance (Batch Sizes)
| Batch Size | Train Loss | Edit Distance | Match Accuracy |
|------------|-----------|---------------|----------------|
| 4          | 0.2046    | 0.5223        | 0.7716         |
| 8          | 0.2776    | 0.7014        | 0.7090         |
| 16         | 0.1784    | 0.4678        | 0.7798         |
| 32         | 0.2889    | 0.7244        | 0.6860         |
| 64         | 0.3106    | 0.7617        | 0.6682         |

### Validation Performance (Batch Sizes)
| Batch Size | Val Loss | Edit Distance | Match Accuracy |
|------------|----------|---------------|----------------|
| 4          | 0.8397   | 1.6775        | 0.5038         |
| 8          | 0.7930   | 1.5747        | 0.4970         |
| 16         | 0.9389   | 1.6759        | 0.4689         |
| 32         | 1.0829   | 2.0439        | 0.3915         |
| 64         | 1.0366   | 2.0638        | 0.3794         |

**Batch size = 4 achieved the best results** with lowest edit distance and highest validation accuracy.  

---
