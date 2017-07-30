# Histogram Of Oriented Gradients (HOG)
HOG method is one of the famous techniques for object recognition and edge detection. This method has been proposed by N. Dalal and B. Triggs in their research paper - "Histograms of Oriented Gradients for Human Detection, CVPR, 2005". The method is quite simple to devise and has been first experimented for human detection (that is, pedestrians on roads). Soon, this technique took it's way to the detection of other objects. The method has the reputation of achieving upto 98 % accuracy for human detection. In a paper written by Robert Arroyo and Miguel Angel Sotelo (the paper can be found at https://www.researchgate.net/publication/260351106 ), it has been mentioned that use of HOG along with SVM classifier fetches an accuracy upto 93% for car logo recognition.

The code is written in PYTHON and TENSORFLOW

## Getting Started - Feature Extraction Using HOG
The HOG descriptor's code uploaded here, is for classification of car logos.
Hog descriptor uses edge detection by gradient calculation and histograms of gradients, with magnitudes as weights.
The code uses [-1 0 -1] kernel for gradient magnitude and orientation calculation. Gradients are calculated in the range [0,180]. Histograms of 8 bins are calculated with magnitudes as weights. Each image is checked if its of 32X32 size, else its resized. The code reads images in greyscale.
The images are normalised for gamma, and then, for normal contrast. Each 32X32 image pixel matrix, is organised into 8X8 cells and then, histograms are calculated for each cell. Then, a 4X4 matrix with 8 bins in each cell is obtained. This matrix is organised as 2X2 blocks(with 50% overlap) and normalised, by dividing with the magnitude of histogram bins' vector. A total of 9 blocks X 4 cells X 8 bins  = 288 features
Then, these features are extracted for each image and trained with an SVM. The accuracy obtained for the dataset uploaded in the logos.zip folder, containing about 60 images for each of 14 brand logos, is 88%.

### Prerequisites
One may need python (any version above 3), tensorflow, numpy, Pillow, and matplotlib.

### Installing
Follow the following links for installing the prerequisites:
1) Python 3.5 : https://www.python.org/downloads/release/python-353/
2) Tensorflow : https://www.tensorflow.org/install/
3) Pillow : https://pillow.readthedocs.io/en/4.2.x/installation.html#basic-installation
4) Numpy : https://docs.scipy.org/doc/numpy-1.10.1/user/install.html
5) matplotlib : https://matplotlib.org/faq/installing_faq.html

## Browsing the folder
The folder contains a zip file called logos.zip, which contains the data set for training the HOG-SVM. This can be downloaded and extracted to the folder 'new_train_data' (make folder if not present...)

Use the code 'prepare_data.py' to prepare data (if data from logos.zip is not being used)
Use the code 'hog.py' to extract hog features.
Use the code 'train.py' to train Linear SVM or Logistic Classifier

### hog.py
Contains the required functions for extracting histograms from images after pre-processing. Run the following code:

```
hog.py
```

### train.py
Extract the downloaded data, into the folder - 'new_train_data' in current folder < OR > If the data is not of logos as downloaded, make the data folders classwise as needed.
First run the other code - 'prepare_data.py'; for this, create the data as needed in the folder - 'raw_train_data', in current folder. The main 'raw_data_folder', must contain sub-folders of classes (ex: BRANDS), then each of these folders must contain sub-folders of labels (ex: MODELS), and then the relevant data must be in the respective folders. The data for training will be saved in the 'new_train_data'
The default paths can be changed in the beginning of each code. But, this is not recommended
Run the train.py code as follows:

1) For training data :
```
train.py train
```
When prompted, for Logistic Classification: enter 'LOGIST'; for Linear SVM MulitClass classification : enter 'SVM'

2) For classifying based on folder names in the 'new_train_data' :
```
train.py classify
```
When prompted, for Logistic Classification: enter 'LOGIST';	for Linear SVM MulitClass classification : enter 'SVM'

## Authors

* **Sunkari Preetham Paul** - [Impetuors]
This has been created as a part of the project, with the team : Impetuors (Sunkari Preetham Paul, Praphul Kumar, Harshit Saini, Aadrish Sharma)

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE.md](LICENSE.md) file for details
