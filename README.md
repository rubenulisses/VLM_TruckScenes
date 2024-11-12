# MAN TruckScenes Dataset Evaluation using LLaVA Model

This repository is dedicated to evaluating ground truth data from the MAN TruckScenes dataset using a Vision-Language Model (VLM), specifically the LLaVA-v1.6-Mistral-7B-HF model. This project provides code, configurations, and documentation for data processing, model setup, and iterative evaluation processes across various phases.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Uploading Dataset to Databricks](#uploading-dataset-to-databricks)
- [References](#references)
- [Contact](#contact)

## Project Overview

This project involves evaluating scenes from the MAN TruckScenes dataset using the LLaVA VLM model to analyze differences in ground truth attributes such as weather, area, daytime, season, lighting, and structure. The goal is to enhance the accuracy of scene interpretations by leveraging the capabilities of VLMs. 

## Dataset

The MAN TruckScenes dataset contains annotated scenes specifically designed for testing autonomous driving and machine vision models. We use the [TruckScenes DevKit](https://github.com/TUMFTM/truckscenes-devkit?tab=readme-ov-file) to manage and access this dataset.

For more details about the dataset, see:
- [TruckScenes DevKit on GitHub](https://github.com/TUMFTM/truckscenes-devkit?tab=readme-ov-file)
- [TruckScenes DevKit on PyPI](https://pypi.org/project/truckscenes-devkit/)

## Model

The model utilized in this project is the LLaVA-v1.6-Mistral-7B-HF, a VLM designed for scene evaluation tasks. This model is hosted on [Hugging Face](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) where you can find details on how to set up and configure it.

## Installation

### Requirements
To run this project, install the following dependencies:

```plaintext
torch
transformers
truckscenes-devkit
opencv-python
pandas
```

Install these using `pip`:
```bash
pip install -r requirements.txt
```

### Databricks Setup

If you are using Databricks, ensure you configure the Databricks environment with the appropriate node type and Databricks runtime version.

## Usage

1. **Data Loading**: Load and preprocess the TruckScenes dataset.
2. **Model Initialization**: Initialize the LLaVA model with specific configurations.
3. **Evaluation**: Run each phase of the evaluation as described in `project.ipynb`.

## Uploading Dataset to Databricks

Since the MAN TruckScenes dataset is large, we will split it into smaller chunks and upload it to Databricks using the Databricks CLI.

### Step 1: Install Databricks CLI

First, install the Databricks CLI if you haven't already:

```bash
pip install databricks-cli
```

Configure the CLI with your Databricks token:
```bash
databricks configure --token
```

### Step 2: Split the Dataset into Smaller Chunks

If the dataset is provided in a large ZIP file, split it into smaller chunks to facilitate upload. You can use the following command (for Linux/MacOS):

```bash
split -b 500M your_dataset.zip dataset_chunk_
```

This command will split `your_dataset.zip` into chunks of 500 MB each, labeled as `dataset_chunk_aa`, `dataset_chunk_ab`, and so on.

### Step 3: Upload to DBFS

Use the following command to upload each chunk to Databricks DBFS (Databricks File System):

```bash
databricks fs cp dataset_chunk_aa dbfs:/path/to/destination/ --overwrite
databricks fs cp dataset_chunk_ab dbfs:/path/to/destination/ --overwrite
# Repeat for all chunks
```

Once all chunks are uploaded, you can combine them within Databricks if needed.

## Hugging Face Setup

To use the pre-trained LLaVA model from Hugging Face, you need to log in with your Hugging Face account and access token.

### Step 1: Get Hugging Face Token

1. Go to [Hugging Face](https://huggingface.co) and log in to your account.
2. Generate an access token from your [account settings](https://huggingface.co/settings/tokens).
3. Copy this token for later use.

### Step 2: Log in to Hugging Face in the Notebook

In your notebook, run the following code to log in to Hugging Face:

```python
from huggingface_hub import login

# Use your Hugging Face token here
login(token="your_hugging_face_token")
```

## References

- [TruckScenes DevKit on GitHub](https://github.com/TUMFTM/truckscenes-devkit?tab=readme-ov-file)
- [TruckScenes DevKit on PyPI](https://pypi.org/project/truckscenes-devkit/)
- [TruckScenes DevKit Paper](https://arxiv.org/pdf/2407.07462)
- [LLaVA GitHub Repository](https://github.com/haotian-liu/LLaVA)
- [LLaVA Paper](https://arxiv.org/pdf/2304.08485)

## Contact

For any inquiries, feel free to reach out:

- **Email**: [ruben.fonseca@scania.com](mailto:ruben.fonseca@scania.com)
- **LinkedIn**: [Ruben Fonseca](https://www.linkedin.com/in/ruben-fonseca-56643a170/)
