# Segmentation of Radial Nerve in Ultrasound Images Using Convolutional Neural Network
## Abstract
In recent years, deep learning has achieved huge successes in many aspects such as object recognition and semantic analysis. Deep learning gets a great development in medical image as well. Peripheral Nerve Blocks is a type of regional anesthesia which need to find out the location of nerves and inject anesthetic nearby using ultrasound scanning. However, images recognition by medical experts is time-consuming. We trained a deep learning model which can perform segmentation of nerves in real time processing. It can be applied to an anesthesia technique called ‘Ultrasound-Guided Regional Anesthesia (UGRA). UGRA has less side-effect compare to general anesthesia. Our training dataset are acquired from two collaborative hospitals, Kaohsiung Veterans General Hospital (KVGH) and Kaohsiung Medical University Chung-Ho Memorial Hospital (KMUH). Our frameworks are based on the U-net model. Moreover, we prune our models using the Net-Trim algorithm, which is capable to reduce the parameters from a trained model. A simplified model consumes less prediction time and memory space. Our proposal achieves 0.59 Dice Coefficient and 0.93 accuracy for nerve segmentation.
## Method
![image](https://github.com/asd5637/UltrasoundNerveSegmentation/blob/master/images/method.png)
##### Figure 1: The overview of our method
### Pre-processing (Wiener filter)
Ultrasound imaging always encounters speckle noise problem. We utilized wiener filter to reduce noise in the images. Wiener filter is based on both global statistics and local statistics.
![image](https://github.com/asd5637/UltrasoundNerveSegmentation/blob/master/images/wiener.png)
##### Figure 2: Wiener filtered images with different kernel size using wiener filter. The upper left image is the input source data. The others are filtered by 3x3, 5x5 and 7x7 kernel separately.
## Ensemble modeling
In the beginning, we tried to segment nerves with a trimmed 5-layer U-net by applying Net-Trim algorithm. However, we found that the prediction results contained more false negative case, which lead to a lower accuracy rate. As we tried 3 layers and 4 layers U-net in the same settings, we observed that with a shallower model (i.e. less layers) could gain higher accuracy rates but a lower dice coefficient score.The false negative cases had decreased a lot in models with fewer layers. In order to increase accuracy rate without losing dice score, we used ensemble modeling based on these three models. The final prediction results were determined by average voting from the models.
![image](https://github.com/asd5637/UltrasoundNerveSegmentation/blob/master/images/model.png)

