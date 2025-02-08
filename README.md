# CNN Model for Plastic Waste Classification
## Overview
This project focuses on building a Convolutional Neural Network (CNN) model to classify images of plastic waste into various categories. The primary goal is to enhance waste management systems by improving the segregation and recycling process using deep learning technologies.

## Table of Contents
- Project Description
- Dataset
- Model design
- Training
- Weekly Progress
- Technologies Used
- Future Scope
- Contributing


## Project Description
Plastic pollution is a growing concern globally, and effective waste segregation is critical to tackling this issue. This project employs a CNN model to classify plastic waste into distinct categories, facilitating automated waste management.

## Dataset
The dataset used for this project is the Waste Classification Data by Sashaank Sekar. It contains a total of 25,077 labeled images, divided into two categories: Organic and Recyclable. This dataset is designed to facilitate waste classification tasks using machine learning techniques.

Key Details:
-Total Images: 25,077
-Training Data: 22,564 images (85%)
-Test Data: 2,513 images (15%)
-Classes: Organic and Recyclable
-Purpose: To aid in automating waste management and reducing the environmental impact of improper waste disposal.
-Approach:
-Studied waste management strategies and white papers.
-Analyzed the composition of household waste.
-Segregated waste into two categories (Organic and Recyclable).
-Leveraged IoT and machine learning to automate waste classification.

Dataset Link:
You can access the dataset here: https://www.kaggle.com/datasets/techsash/waste-classification-data

Note: Ensure appropriate dataset licensing and usage guidelines are followed.

## Model Design
The project implements a CNN designed to extract features, reduce dimensions, and classify images.

Core Components of the Model:
- Convolutional Layers: Extract essential features from the images.•Pooling Layers: Reduce the dimensionality of features.
- Fully Connected Layers: Perform the final classification.
- Activation Functions: Utilize ReLU and Softmax functions.
A detailed diagram of the CNN architecture will be added later.

## Training Process
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: Configurable (default: 25)
- Batch Size: Configurable (default: 32)
- Data Augmentation: Used to improve generalization and model performance.

## Weekly Updates
### Week 1 (20th–27th January 2025):
- Imported necessary libraries and frameworks.
- Set up the project environment.
- Explored and understood the dataset structure.

### Week 2:
- Successfully trained the waste classification model using CNN.
- Evaluated the model's performance by analyzing accuracy and loss graphs.
- Fine-tuned hyperparameters to improve accuracy.
- Tested the model with validation data to ensure reliable classification results..

### Week 3:
- Successfully deployed the waste classification model as a public web app using Streamlit Cloud.
- Ensured seamless integration of the trained CNN model into the web application.
- Troubleshot and resolved deployment issues (dependency management, TensorFlow setup).
- Generated a publicly accessible link for real-time classification of images.
- Prepared a detailed project report covering model architecture, training results, and deployment.
- Created a PowerPoint presentation (PPT) summarizing the project workflow and findings.
- The link of website is https://cnnplasticwasteclassification-npcccewwuatmyjyduxema8.streamlit.app/

## Technologies Employed
- -Python
- TensorFlow/Keras
- OpenCV
- NumPy
-Pandas
- Matplotlib


## Future Enhancements
- Extend the dataset to include additional plastic waste categories.
- Develop a web or mobile application for real-time waste classification.
- Integrate the system with IoT-enabled waste management solutions.

## Contributing
We welcome contributions! If you'd like to contribute to this project, feel free to open an issue or submit a pull request.






