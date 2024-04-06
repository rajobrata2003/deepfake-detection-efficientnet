# Deepfake Detection
This repository provides code for detecting deepfake images and videos using pre-trained models.
## Requirements
Run pip install -r requirements.txt to install all the required packages
## Usage
Run the webui code depending on your requirement (Image or Video), upload the required media, and click on Submit
Prediction will be generated accordingly.

## Details
EfficientNet was used for the base feature extraction.
A small portion of the FaceForensics++ dataset was used.
Only around 8000 images were used for detection. Accuracy can be improved using more images and more variety.
