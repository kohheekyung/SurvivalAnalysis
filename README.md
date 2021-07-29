# Survival Analysis
This project is to perform survial analysis by deep learning method. This could be an alternative to Kaplan-Meier estimator(https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator).
Survival functions which measure the conversion time to alzheimer disease of patients were predicted.

The dataset is based on open dataset ADNI.
A csv file of ADNI contains information of patients (Demography, Clinical Data, Convert Time).
Detailed information can be found in http://adni.loni.usc.edu/methods/documents/.

## Missing data problem - GAIN
The ADNI dataset have missing values(features not label). To solve the problems, the missing values were imputed by generative model.
The GAIN (Generative Adversarial Imputation Networks) is known to solve missing value. 

Why missing value problem happens
- It is hard to measure all diagnosis procedures.
- Some patients can't complete all clinical tests due to their health status.
- Sometimes, patients could refuse tests such as biopsy, lumbar puncture or even blood test.
- Standard medical test procedure can change over time.

Network architecture and details are described in the paper (https://arxiv.org/pdf/1806.02920.pdf).
Jinsung Yoon, 2018, GAIN: Missing Data Imputation using Generative Adversarial Nets, ICML.

### Implementation Details
1. Data preprocessing of ADNI
-	Min/Max Normalization each feature range of [0, 1]
-	Encode string-value features to int-value label
<img src="https://github.com/kohheekyung/SurvivalAnalysis/blob/main/resources/GAIN_preprocessing.png" width="300">

2. Generator
- Input  - (1) Real data matrix (2)  Random vector at Nan-value (3) Mask matrix
- output - Imputed data
- Loss -  (1) Mean Square Error ( Real data & Imputed data) (2) Adversarial loss

3. Discriminator 
- Input - Imputed data from generator
- Output - Mask vector  [ real (1), imputed (0), confusing data (0.5) ]
- Hint mask  - help training of discriminator 
- Loss - Cross entropy 

### Imputation result
RMSE is model evaluation metric
<img src="https://github.com/kohheekyung/SurvivalAnalysis/blob/main/resources/rmse.png">

<img src="https://github.com/kohheekyung/SurvivalAnalysis/blob/main/resources/imputation.png" align ="center" width="800">

## Survival analysis - DL based
1. Data preprocessing of ADNI for survial analysis
- Standardize numerical feautures
- Encode string-value features to int-value label
- Categorize target time to multi label
<img src="https://github.com/kohheekyung/SurvivalAnalysis/blob/main/resources/mlp_preprocessing.png" width="300">

2. Model
- Multi layer perceptron with three hidden layer, RELU activation
- Loss - categorical crossentropy

### Comparison with Kaplan-Meier based survival function
<img src="https://github.com/kohheekyung/SurvivalAnalysis/blob/main/resources/result.png" align ="center" width="700">

### Feature Importance
This is training history of DL based survival analysis.

<img src="https://github.com/kohheekyung/SurvivalAnalysis/blob/main/resources/loss.png" align="center" width="700">

If remove important features such as 'EcogPtLang', 'EcogPtLang_bl', training is not progressive.

### Prerequisite
- tensorflow-gpu 1.15.0
- keras 2.3.1
- nibabel
