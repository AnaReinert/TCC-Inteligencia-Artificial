This project aims to develop a system for detecting workers with and without helmets, using computer vision and machine learning techniques. The process begins with the collection of a dataset of images, including heads with and without helmets, obtained through the Roboflow website. Next, two facial detection algorithms were analyzed: SSD-MobileNet and YOLOV8, both undergoing a proof of concept and training. The model with the best performance, based on precision metrics, was chosen for the next phase.

After selecting the ideal model, facial recognition was implemented using the face-recognition library, which allows for the identification of faces even at long distances or from different perspectives. Finally, for sending notifications to the site manager, the SMTPLIB library was used to send emails whenever a worker without a helmet is identified, containing the name, date, and time of the recognition.

Below, the stages of the project development will be briefly outlined:

**1. Choosing the architectures:**

Due to the nature of the project, it was necessary to find object detection architectures that were agile enough for real-time identification of workers without helmets. For this, two object detection algorithms that use Single-Shot architecture were chosen: YOLO and SSD.

**2. Preparing the dataset:**

In the first stage, for the detection of workers to be successful, it was necessary to collect images of workers with and without helmets. There are various approaches to this collection, but for this project, images of heads with and without helmets were collected, with helmets of various colors, and in both samples, people of different ethnicities, phenotypes, and genders were used.

The second stage of dataset preparation was the labeling of images and annotation control. Since the images were collected from various online datasets, some had classes that would not be used and some had annotations that were not adapted to what was needed. To assist in this process, the Roboflow platform was used, which provides heat maps and class representation analysis.


![image](https://github.com/user-attachments/assets/7c607f6a-4ebe-4472-a189-cfe103a214bc)

![image](https://github.com/user-attachments/assets/feafe39b-20e5-4aba-bac5-361f7ee57c7c)


**3. Training**

For the training, the clones of the Ultralytics repositories were used for YOLO training and TensorFlow for SSD training. For the versions, YOLOV8 and SSD-MobileNet were used.In the separation of the sets, it was divided into 80% for training and 20%, equally divided, for validation and training.

**4. Analysis of results**
The results of the two models generated a confusion matrix and precision, recall, and accuracy metrics.

_An observation about the generated Confusion Matrix: When there is an incidence of background in the vertical of the matrix, it means that the model incorrectly identified a class in an area without elements, as in the case of YoloV8 identifying a head where there was none. The incidence of background in the horizontal direction indicates that the model failed to identify a class where it should have marked, such as SSD-MobileNet not recognizing a head where it should have._

Before comparing the precision, recall, and accuracy metrics of the YoloV8 and SSD-MobileNet models, it is important to consider the project's focus, which is to identify people without helmets. Although both models perform well in helmet detection, the analysis prioritizes performance in head detection.
- Precision: Measures the proportion of correct predictions among the predicted ones. For heads, YoloV8 had 99.7% and SSD-MobileNet 99.8%. For helmets, YoloV8 achieved 98.2% and SSD-MobileNet 97.9%. YoloV8 had slightly better performance, making fewer errors in helmet predictions.
- Recall: Evaluates the ability to correctly identify the classes present in the images. YoloV8 achieved 97.5% for heads and 99.7% for helmets, while SSD-MobileNet reached 80.4% for heads and 96.7% for helmets. YoloV8 performed better, especially in head detection.

Therefore, YoloV8 showed better performance in recall and precision metrics, making it the most efficient model for the project's focus.

Results SSD-MobileNet
![image](https://github.com/user-attachments/assets/44926a40-8b7d-441c-9c59-d0ed8522244f)
![image](https://github.com/user-attachments/assets/fa3e0622-8c59-4c8d-963f-1f329ba83019)

Results YOLOV8
![image](https://github.com/user-attachments/assets/162fac63-8b42-463c-ab86-45a3e583aa79)
![image](https://github.com/user-attachments/assets/c21141f6-60e6-4f0f-a35d-b657e87236b7)

**5. Project completion:**

To conclude, the winning model was introduced into the final project, which uses the face-recognition library to identify workers without helmets and subsequently sends an email notification to the site manager through the Python SMPT library, which uses the email sending protocol.

![image](https://github.com/user-attachments/assets/e9600441-4744-427c-940d-2aed4c76f422)

![image](https://github.com/user-attachments/assets/f87e09f9-ab84-4762-8bd2-c9f622899272)


