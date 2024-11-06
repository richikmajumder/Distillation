# CIFAR-10 Knowledge Distillation

This repository implements a knowledge distillation approach to train a student model on the CIFAR-10 dataset. The student model learns from a pre-trained teacher model (ResNet-18). The goal is to improve the student model's performance by leveraging the teacher's predictions.

## Dependencies

The following libraries are required to run this code:

- PyTorch
- torchvision
- matplotlib
- seaborn
- scikit-learn

The code uses a pre-trained ResNet-18 model as the teacher, which is fine-tuned for the CIFAR-10 classification task. The student model is a simple CNN with dropout, and the training involves a distillation loss function combining soft and hard losses.

## Installation

To set up the environment, clone this repository and install the dependencies using `pip`:

1. Clone the repository:
   git clone <repository-url>
   cd <repository-directory>
   
2. Create a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install dependencies
   pip install -r requirements.txt

4. Once the environment is set up, you can run the training process with the following command:
   python run.py
