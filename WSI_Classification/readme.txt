In this project we used the Foundation models extracted features for WSI classification (MSI vs MSS/nonMSI). 


We used different approaches for the representation of WSI because we extracted the patch level features from foundational models. 

1. We average all the patches features to make one averaged feature of WSI. 

2. We applied KMeans clustering to make three clusters of each WSI patches and then concatenate/stack all three cluster centroid feature vectors to make WSI representation. 

3. We make nine different groups of patches in each WSI by classifying each patch using a pretrained classifier on Zenodo CRC-100K dataset.
3.1. Then we averaged same tissue type patches features and in this way we got nine different features representation and stack those to make one WSI level representation. 
3.2. After this, we experimented using different tissue type features of WSI and selected two best performing tissue types (MUC and TUM) patches to represent whole WSI representation. 


Using all the above WSI level representation then we trained linear classifiers to classify the MSI and MSS WSIs. We trained and experimented four different linear classifiers. 
A. Linear_probe/ Logistic Regression
B. ANN
C. KNN
D. ProtoNet

