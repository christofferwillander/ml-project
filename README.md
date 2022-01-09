# ml-project
**Final project in the course Machine Learning (DV2578). Comparison of supervised learning algorithms for classification of malicious URLs.**

## General information
Model information, model performance metrics are printed in the terminal after execution of the *url_classifier.py* script based on the generated dataset. Furthermore, a cumulative confusion matrix for each respective model is plotted after execution. In this project the classifiers Random Forest, Gauissian Naive Bayes and K-Nearest Neighbors were used. Hyperparameter tuning was conducted using *Random Search Cross Validation* for a few key parameters per model. Each model was evaluated using 10-fold stratified CV, and the average performance metrics were calculated for every model based on this. The dataset used was generated based on the *ISCX-URL2016* collection from *University of New Brunswick*, which consists of classified URLs. Additional features were extraxted and generated from these URLs (e.g. string & URL characteristics, WHOIS information, etc.)

## Setup
**Requirements:** *Python 3.9.9*, *pip 21.2.4*, *pandas*, *scikit-learn*, *numpy*, *matplotlib*, *python-whois 0.7.2*, *tld*.

Simply install Python and pip, and then run the *pip install -r requirements.txt* command in the terminal to install all the required packages.
**N.B.** Make sure CWD is the root of the assignment folder.

## Script execution
The script is executed by simply typing *python url_classifier.py* in the terminal while the CWD is the root of the assignment folder.
The *feature_extractor.py* script was used to generate *./data/urls.data* (the dataset used) based on *./data/initialData.csv*.

## Author
Christoffer Willander (DVACD17) - *chal17@student.bth.se*
