# Alphabet-Level Indian Sign Language Recognition

## Overview
Effective communication is the cornerstone of human interaction, yet it remains a challenge for deaf and mute individuals due to a lack of widespread understanding of sign language. This project aims to bridge this gap by developing a **real-time Indian Sign Language (ISL) Recognition system** using **deep learning** and **computer vision**. The system interprets ISL gestures and translates them into English, enabling seamless communication.

## Features
- **Custom Dataset Creation**: Leveraged Python and OpenCV to create a dataset of ISL alphabets and numerals captured via webcam.
- **Real-Time Recognition**: Utilized TensorFlow’s Object Detection API and transfer learning to achieve real-time gesture detection.
- **State-of-the-Art Models**: Fine-tuned EfficientDet D0, SSD ResNet101, and SSD MobileNet V2 FPNLite architectures for ISL recognition.
- **High Accuracy**: Achieved an average confidence level of **80%**, demonstrating reliable performance despite a limited dataset size.
- **Scalable and Adaptable**: Designed to accommodate larger datasets and expand to other sign languages with minimal modifications.

## Methodology
1. **Data Acquisition**:
   - Captured images of ISL gestures using OpenCV and Python.
   - Created a well-annotated dataset of 2000 images, covering all ISL alphabets and numerals.
   - Preprocessed the dataset and generated TensorFlow (TF) records for training and validation.

2. **Model Training**:
   - Applied transfer learning on pre-trained object detection models from TensorFlow’s Model Zoo.
   - Configured hyperparameters for optimal performance, including 50,000 training steps.
   - Fine-tuned models to align with the dataset’s 36 classes (alphabets and numerals).

3. **Evaluation**:
   - Tested models on validation data with a 90:10 train-test split.
   - Analyzed confidence levels and identified areas for improvement, particularly increasing dataset diversity.

## Results
- **Real-Time Detection**: The system identifies ISL alphabets and numerals with an average confidence level of **80%**.
- **Model Performance**: Achieved low total loss values:
  - EfficientDet D0: 0.119
  - SSD ResNet101: 0.061
  - SSD MobileNet V2: 0.058

## Future Enhancements
- Expand the dataset to include more diverse ISL gestures and scenarios.
- Explore alternative deep learning architectures for improved accuracy and efficiency.
- Adapt the system to recognize continuous sign language gestures and sentences.
- Extend support for other regional and global sign languages.

## Keywords
Indian Sign Language, Sign Language Recognition, Deep Learning, TensorFlow, Real-Time Detection, Computer Vision

## Acknowledgements
This project is a testament to the power of technology in promoting inclusivity and breaking communication barriers. Inspired by the needs of the deaf and mute community, it underscores the importance of innovative solutions in building a more accessible world.

---


