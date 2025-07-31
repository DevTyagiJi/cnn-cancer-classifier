#  CNN-Based Histopathology Image Classifier

This project uses a Convolutional Neural Network (CNN) to classify histopathology images of breast tissue as **cancerous** or **normal**. Built using TensorFlow and Grad-CAM for visual interpretability.

##  Features
- Binary classification using CNN  
- Image preprocessing & augmentation  

- Easy prediction on custom images

# Dataset 
https://drive.google.com/drive/folders/14g08diXf2nf3y9IV90zSyE9-Z-Z7P80z?usp=sharing

## 🗂 Dataset Structure
histopathology/  
├── cancer/  
└── normal/

##  How to Use
pip install -r requirements.txt  
python train.py         → Train the model  
python predict.py       → Predict a new image  
python gradcam.py       → Generate heatmaps

## Future Scope
- Deploy as a web app  
- Add multi-class support  
- Use pretrained models (Transfer Learning)

## Author
**Dev Tyagi** — B.Tech EE, DTU  
GitHub: *[your-username]*  
Email: *your-email@example.com*

