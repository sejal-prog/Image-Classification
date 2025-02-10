# Image-Classification
Uses ResNet model to classify cat vs dog. 



Project Overview

This project implements a binary image classification model using PyTorch to distinguish between cats and dogs. The model is based on a modified ResNet architecture trained on a dataset of cat and dog images.


📂 Dataset & Preprocessing

Dataset

We use the Animal Dataset from Kaggle.

The dataset contains images of cats, dogs, and horses, but we ignore the "horses" category and only focus on cats vs. dogs.

Preprocessing Pipeline

Resizing: All images are resized to 64x64 pixels (CNNs require fixed-size input).

Grayscale Conversion: Since color is not crucial for this task, we convert all images to grayscale (1 channel instead of 3).

Normalization: We normalize pixel values to [-1,1] to improve training stability.

Data Augmentation: We apply random horizontal flips and rotations to improve generalization.


🏗️ Model Development

We use a modified ResNet architecture with:

Input layer adjusted for grayscale (1 channel instead of 3)

Final layer modified for binary classification (2 output neurons: Cat vs. Dog)

Skip connections to help with gradient flow and prevent vanishing gradients


🏋️‍♂️ Training

Loss Function: CrossEntropyLoss 

Optimizer: Adam 

Training Logs: TensorBoard is used to log loss curves

Checkpoints: The best model is saved automatically

Run Training 

- python train.py

Training logs are saved in logs/

The best model is saved in models/best_model.pth

🔍 Model Testing & Inference

We provide a script (inference.py) that loads the trained model and predicts whether a given image is a cat or a dog.

Run Inference

- python inference.py --image Dataset/test/dog/sample.jpg

The output will display the predicted label and confidence score.

📊 Model Evaluation

The model is evaluated on a separate test set.

Metrics used:

Accuracy: Measures the percentage of correct predictions.

Loss Curve: Plotted using TensorBoard and added images in Curve folder

Interpretation - 
A low training and validation loss is a good sign, but since it approaches near-zero loss, the model might be overfitting. 

📦 Installation & Setup

Step 1: Install Dependencies

- pip install -r requirements.txt

Step 2: Train the Model

- python train.py

Step 3: Run Predictions

- python inference.py --image Dataset/test/dog/sample.jpg


Challenges Faced:

This was my first time developing a model entirely on my own, as I had previously always worked in a team. It was a fun and rewarding experience, but I did face some challenges along the way:

1. Input Channel Mismatch: The ResNet model is designed to work with RGB channels, but I had already converted my images to grayscale, resulting in errors. I eventually adjusted my approach by ensuring that the CNN could accept a single channel as input, which resolved the issue.

2. Models seems to overfit. Possible solutions would be - Early stopping, use regularization techniques. 
For Larger Datasets:

Leveraging Pretrained Models: To handle larger datasets, one effective strategy is to use pretrained models. This allows the model to build upon features learned from a broader, more generalized dataset.

Advanced Optimizers and Learning Rate Schedules: Incorporating advanced optimizers and dynamic learning rate schedules can help the model converge more quickly .

Hyperparameter Tuning: By fine-tuning parameters such as the learning rate, batch size, and number of layers, we can maximize the model’s performance and achieve optimal results.
>>>>>>> 3e18463 (Initial commit)
