# Air Traffic Controller (ATC) Fatigue Detection via Tri-Domain Adversarial CRNN

## Problem Statement
Air Traffic Control Operators (ATCo) operate in high-stress, safety-critical environments where cognitive fatigue can lead to catastrophic incidents. While voice communications provide a continuous, non-intrusive stream of biometric data to monitor fatigue, utilizing this data introduces two major operational hurdles: 
1. **Privacy:** Voice features inherently leak speaker identity, raising surveillance and privacy concerns for operators.
2. **Environmental Variance:** ATC audio is subject to severe, dynamic hardware degradation (e.g., DSP echo, signal fade, tower hum), making standard models brittle in real-world deployments.

## Project Overview
This repository contains an end-to-end machine learning and deep learning pipeline that solves these challenges. We designed a **Multi-Domain Adversarial Network (MDAN)** utilizing a Convolutional Recurrent Neural Network (CRNN). By incorporating a Gradient Reversal Layer (GRL), the model maximizes fatigue detection accuracy while actively penalizing the network for retaining speaker identity or hardware-specific acoustics. 

The resulting architecture achieves ~99% accuracy on the main fatigue task while maintaining near-random guessing for speaker identification (18.10%) and radio identification (24.11%), ensuring both operator privacy and operational resilience. Finally, the frame-level inferences are translated into an actionable Continuous Fatigue Index (CFI) dashboard for dynamic shift scheduling.

---

## Repository Structure & File Analysis

### 1. Data Preprocessing & Spatial Feature Extraction (`01-data-prep-and-feature-extraction.ipynb`)
This module handles the transformation of raw `.wav` audio files into continuous spatial tensors.
* **Continuous Sample Builder**: Randomly samples and stitches fatigue and non-fatigue audio clips to create 5-minute continuous sequences simulating real ATC communication flows.
* **Windowing & Labeling**: Slices the continuous audio into 2.0-second windows (16,000 samples at 8000 Hz) and assigns labels based on the dominant class overlap (Ambiguous, Fatigue, Non-Fatigue).
* **2D Spectrogram Extraction**: Utilizes GPU-accelerated `torchaudio.transforms.MelSpectrogram` (512 n_fft, 128 n_mels) to convert 1D audio tensors into 2D log-mel spectrograms.
* **Data Routing**: Automatically splits the dataset (70% Train, 15% Validation, 15% Test) and securely uploads the processed 18k spatial features to Kaggle Datasets for downstream training.

### 2. Baseline Machine Learning Models (`02-baseline-models.ipynb`)
Provides classical machine learning benchmarks using acoustic features to compare against our deep learning approach.
* **Feature Extraction**: Implements OpenSMILE to extract the `eGeMAPSv02` feature set from the raw audio files.
* **Data Pipeline**: Uses `SimpleImputer` for missing values and `StandardScaler` for feature scaling.
* **Model Training**: Trains four distinct classifiers to establish baseline fatigue detection accuracy:
  * **XGBoost**: ~80.07% Accuracy
  * **LightGBM**: ~80.26% Accuracy
  * **Random Forest**: ~80.21% Accuracy
  * **Support Vector Machine (SVM)**: ~83.71% Accuracy
* **Evaluation**: Outputs classification reports, confusion matrices, and ROC curves for each baseline.

### 3. Tri-Domain CRNN MDAN Training (`03-mdan-model-training-and-evaluation.ipynb`)
The core deep learning engine that trains the CRNN with a Gradient Reversal Layer (GRL).
* **Architecture**:
  * **Backbone**: A 3-layer 2D CNN (extracts spatial features) followed by a 2-layer Bidirectional LSTM (captures temporal dependencies).
  * **Utility Head**: Classifies operator state into 3 fatigue classes.
  * **Privacy Adversary Head**: Predicts speaker identity (7 classes) to enforce subject-agnostic feature learning.
  * **Resilience Adversary Head**: Predicts radio profiles (4 classes) to ensure robustness against environmental noise.
* **On-The-Fly Augmentation**: Applies dynamic radio degradation during training, including DSP echo/reverberation, signal fade/dropout, and cockpit/tower hum.
* **Adversarial Training**: Implements a Gradient Reversal Function with a dynamic alpha scaling schedule (warmup and ramp phases) to penalize the model for retaining speaker and radio-specific information.

### 4. Continuous Fatigue Index (CFI) Inference & Scheduling (`04-cfi-and-dynamic-scheduling.ipynb`)
Translates frame-level model inferences into an actionable operational dashboard.
* **Footprint Extraction**: Evaluates the trained model on validation data to extract true probability footprints for 'Alert' and 'Fatigue' states.
* **CFI Calculation**: Simulates a 5-hour shift using stochastic blending of acoustic states. Computes the instantaneous severity score and applies a Biological Exponential Moving Average (EMA) with an alpha of 0.95 to track accumulated sleep pressure.
* **Dynamic Scheduling Simulation**: Compares airspace demand ($TC_t$) against unmitigated and mitigated supply capacity ($COC_t$). Triggers tactical alerts and automated ATCo handovers when the safety threshold ($\rho > 0.85$) is breached.
* **Dashboard Generation**: Outputs a 3-panel ablation study (`RealModel_Scheduling_Ablation.png`) visualizing the fatigue index, the unmitigated operational risk, and the successfully mitigated bio-dynamic schedule.

---

## Technical Stack

| Category | Technologies / Libraries |
| :--- | :--- |
| **Deep Learning** | PyTorch, Torchaudio |
| **Machine Learning** | XGBoost, LightGBM, Scikit-Learn, Random Forest, SVM |
| **Feature Extraction** | OpenSMILE (eGeMAPSv02) |
| **Data Processing** | NumPy, Pandas, SciPy (`savgol_filter`) |
| **Visualization** | Matplotlib, Seaborn, t-SNE |

---

## Kaggle Execution Guide

This pipeline is optimized for Kaggle's cloud environment. To run these notebooks successfully, follow this step-by-step sequential guide.

### Step 1: Dataset Acquisition & Kaggle Upload
1. **Download Raw Data:** Navigate to the IEEE DataPort repository: [Air Traffic Controller Radiotelephony Fatigue Corpus](https://ieee-dataport.org/documents/air-traffic-controller-radiotelephony-fatigue-corpus).
2. Log in or create a free IEEE account to access the files, and download the dataset to your local machine.
3. **Compress:** If the downloaded files are not already zipped, compress the raw audio directory and its associated labels into a single `.zip` archive.
4. **Upload to Kaggle:** Log in to [Kaggle](https://www.kaggle.com/), go to **Datasets** -> **New Dataset**. Upload your `.zip` file, give it a recognizable title (e.g., "ATC Radiotelephony Fatigue Corpus"), and click **Create**. Keep it private or public based on your preference.

### Step 2: Notebook Import
1. Download the four core notebook files (`.ipynb`) from this repository to your local machine.
2. On Kaggle, navigate to **Create** -> **New Notebook**.
3. Go to **File** -> **Import Notebook** and upload `01-data-prep-and-feature-extraction.ipynb`. Repeat this process in separate browser tabs for `02-baseline-models.ipynb`, `03-mdan-model-training-and-evaluation.ipynb`, and `04-cfi-and-dynamic-scheduling.ipynb`.

### Step 3: Environment Configuration
1. **Enable Internet:** In the right-hand Kaggle sidebar of each notebook, ensure **Internet** is toggled **On** (required for `pip install` commands).
2. **Attach Data:** Click **+ Add Data** in the right sidebar. Search for the Kaggle Dataset you created in Step 1 (e.g., "ATC Radiotelephony Fatigue Corpus") and attach it to your workspace.
3. **Hardware Acceleration:** 
   * For `01-data-prep-and-feature-extraction.ipynb` and `03-mdan-model-training-and-evaluation.ipynb`, go to the **Session Options** (three dots in the top right) -> **Accelerator** and select **GPU T4 x2** or **GPU P100**. Deep learning and spatial feature extraction require GPU acceleration.
   * `02-baseline-models.ipynb` and `04-cfi-and-dynamic-scheduling.ipynb` can be run on a standard CPU environment.

### Step 4: Sequential Execution Order
You **must** run the notebooks in the following exact order, as downstream scripts depend on the outputs of upstream scripts.

1. **Run `01-data-prep-and-feature-extraction.ipynb` first.** 
   * *Action:* Verify the dataset paths point to your attached dataset, then click "Run All". 
   * *Outcome:* This will process the raw audio, generate the Mel-spectrogram tensors, and output the processed dataset. You must save this notebook's output as a *new* Kaggle Dataset so it can be ingested by the next steps.
2. **Run `02-baseline-models.ipynb` (Optional but recommended).**
   * *Action:* Ensure your raw audio dataset is connected and click "Run All".
   * *Outcome:* Generates the OpenSMILE features and logs the baseline ML metrics.
3. **Run `03-mdan-model-training-and-evaluation.ipynb`.**
   * *Action:* Click **+ Add Data**, search for the "Your Work" tab, and attach the *processed spatial tensor dataset* you generated in Step 4.1. Click "Run All".
   * *Outcome:* Trains the adversarial network, generates the `.pth` model weights, and plots the training metrics.
4. **Run `04-cfi-and-dynamic-scheduling.ipynb`.**
   * *Action:* Ensure the trained model weights from Step 4.3 are accessible (you may need to add the output of the training notebook as data). Click "Run All".
   * *Outcome:* Simulates the operational shift and generates the final Continuous Fatigue Index (CFI) scheduling visualizations.

---

## Local Setup & Installation

If you prefer to run the environment locally, install the required dependencies:

```bash
pip install torch torchaudio numpy pandas matplotlib seaborn scipy tqdm
pip install opensmile lightgbm xgboost scikit-learn thop
```