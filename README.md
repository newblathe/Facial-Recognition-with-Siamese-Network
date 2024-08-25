# **One-Shot Facial Verification Model Using Siamese Network**

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Importing the Dependencies](#2-importing-the-dependencies)  
   1. [OpenCV (cv2)](#21-opencv-cv2)  
   2. [OS Module (os)](22-os-module-os)  
   3. [NumPy (np)](#23-numpy-np)  
   4. [Random (random)](#24-random-random)  
   5. [Matplotlib (matplotlib.pyplot)](#25-matplotlib-matplotlib.pyplot)  
   6. [TensorFlow and Keras (tensorflow.keras)](26-tensorflow-and-keras-tensorflow.keras)  
   7. [UUID (uuid)](#2.7-uuid-(uuid))  
3. [Dataset Overview](#3.-dataset-overview)  
   1. [Moving LFW Images to Negatives](#3.1-moving-lfw-images-to-negatives-(section-5.1))  
   2. [Capturing Images via Webcam](#3.2-capturing-images-via-webcam-(section-5.2))  
   3. [Image Augmentation](#3.3-image-augmentation-(section-5.3)-(used-in-“siamese\_modelv2.h5”))  
   4. [Adjusting Brightness and Augmenting Images](#3.4-adjusting-brightness-and-augmenting-images-(section-5.3))  
4. [Dataset Preparation and Processing](#4.-dataset-preparation-and-processing)  
   1. [Dataset Creation](#4.1-dataset-creation-(section-6.1))  
   2. [Preprocessing Function](#4.2-preprocessing-function-(section-6.2))  
   3. [Dataset Labeling](#4.3-dataset-labeling-(section-6.3))  
   4. [Dataset Preparation](#4.4-dataset-preparation-(section-6.4))  
5. [Model Overview](#5.-model-overview)  
   1. [Embedding Model](#5.1-embedding-model-(section-7.1))  
   2. [L1Dist Layer](#5.2-l1dist-layer-(section-7.2))  
   3. [Siamese Model](#5.3-siamese-model-(section-7.3))  
6. [Training the Model](#6.-training-the-model)  
   1. [Components](#6.1-components)  
      * Loss Function and Optimizer  
      * Checkpoints  
      * Gradient Calculation  
   2. [Training Loop](#6.2-training-loop-(section-8.4))  
7. [Evaluating the Model](#7.-evaluating-the-model)  
   1. [Workflow](#7.1-workflow)  
8. [Saving the Model](#8.-saving-the-model)  
9. [Image Verification](#9.-image-verification)  
   1. [Verification Function](#9.1-verification-function-(section-11.1))  
   2. [Webcam Capture and Verification](#9.2-webcam-capture-and-verification-(section-11.2))

10\.  [Accessing and Training the mode](\#10.-accessing-and-training-the-model)l

## **1. Introduction**

This repository contains the implementation of a one-shot facial verification model using a Siamese network. The model is designed to verify whether two facial images belong to the same person.

## **2. Importing the Dependencies**

**Sections 1 and 2** deal with importing the necessary dependencies:

### **2.1 OpenCV (cv2)**

* **Purpose**:  
  * Capture video from the webcam.  
  * Handle image processing tasks such as saving images, cropping frames, and displaying them in a window.

### **2.2 OS Module (os)**

* **Purpose**:  
  * Interact with the file system, including creating and navigating directories and managing file paths.

### **2.3 NumPy (np)** {#2.3-numpy-(np)}

* **Purpose**:  
  * Perform numerical operations on arrays, such as manipulating image data and handling predictions.

### **2.4 Random (random)** {#2.4-random-(random)}

* **Purpose**:  
  * Generate random numbers, useful for any randomization tasks, though it’s not explicitly used in the current script.

### **2.5 Matplotlib (matplotlib.pyplot)** {#2.5-matplotlib-(matplotlib.pyplot)}

* **Purpose**:  
  * Plot and visualize data, though it's included here for potential use in displaying images or results.

### **2.6 TensorFlow and Keras (tensorflow.keras)** {#2.6-tensorflow-and-keras-(tensorflow.keras)}

* **Purpose**:  
  * Build and train deep learning models.  
  * Provides tools like Model, Layer, and various neural network layers (Conv2D, MaxPooling2D, Flatten, Dense, Input) for constructing the Siamese network.

### **2.7 UUID (uuid)** {#2.7-uuid-(uuid)}

* **Purpose**:  
  * Generate unique identifiers, specifically used here to create unique names for saved images.

### **Additional Sections:**

* **Section 3**: Limits GPU growth.  
* **Section 4**: Deals with making the file structure.

## **3**. **Dataset Overview** {#3.-dataset-overview}

This section details how the dataset was created, including capturing, augmenting, and processing images for training the one-shot facial verification model.

#### **3.1 Moving LFW Images to Negatives (Section 5.1)** {#3.1-moving-lfw-images-to-negatives-(section-5.1)}

* The Labeled Faces in the Wild (LFW) dataset is extracted from a tarball file (`lfw.tgz`), and all images are moved to the negatives folder (`NEG_PATH`).

#### **3.2 Capturing Images via Webcam (Section 5.2)** {#3.2-capturing-images-via-webcam-(section-5.2)}

* **Video Capture**: Images are captured using the system's webcam. The captured frames are processed using Haar Cascade classifiers to detect faces. The detected face regions are then cropped and resized to 250x250 pixels.  
* **Capturing Anchors**:  
  * Pressing "a" saves the current frame as an anchor image in the anchor directory (`ANC_PATH`).  
* **Capturing Positives**:  
  * Pressing "p" saves the current frame as a positive image in the positives directory (`POS_PATH`).  
* **Exiting**:  
  * Pressing "e" exits the capture loop, releasing the webcam.

#### **3.3 Image Augmentation (Section 5.3) (used in “siamese\_modelv2.h5”)** {#3.3-image-augmentation-(section-5.3)-(used-in-“siamese_modelv2.h5”)}

* **Augmentation Function:**:  
  * An `augment` function is defined to apply various transformations to the images, such as brightness adjustment, horizontal flipping, contrast adjustment, JPEG quality adjustment, and saturation adjustment. Each image undergoes five different augmentations, ensuring diversity in the dataset.

####  **3.4 Adjusting Brightness and Augmenting Images (Section 5.3)** {#3.4-adjusting-brightness-and-augmenting-images-(section-5.3)}

* **Anchor Images**:  
  * Some anchor images have their brightness reduced using TensorFlow's image adjustment functions. These adjusted images are saved back into the anchor directory.  
  * The adjusted images are then augmented using the `augment` function, and the resulting images are saved in the anchor directory.  
* **Positive Images**:  
  * Similar to the anchor images, some positive images have their brightness reduced and are saved back into the positives directory.  
  * These images are then augmented, with the augmented versions saved back into the positives directory.

## **4\. Dataset Preparation and Processing** {#4.-dataset-preparation-and-processing}

The dataset preparation and processing are detailed throughout Section 6\.

### **4.1 Dataset Creation (Section 6.1)** {#4.1-dataset-creation-(section-6.1)}

* **Anchor Images**:  
  * Loaded from `ANC_PATH`, taking 3,684 images and shuffled with a buffer size of 3,000.  
* **Positive Images**:  
  * Loaded from `POS_PATH`, taking 3,684 images and shuffled with a buffer size of 3,000.  
* **Negative Images**:  
  * Loaded from `NEG_PATH`, taking 3,884 images and shuffled with a buffer size of 3,000.

### **4.2 Preprocessing Function (Section 6.2)** {#4.2-preprocessing-function-(section-6.2)}

* **`preprocess(file_path)`**:  
  * Reads, decodes, resizes, and normalizes images to 100x100 pixels and scales pixel values to \[0, 1\].

### **4.3 Dataset Labeling (Section 6.3)** {#4.3-dataset-labeling-(section-6.3)}

* **Positive Pairs**:  
  * Zipped with anchor images and labeled with `1` (indicating similarity).  
* **Negative Pairs**:  
  * Zipped with anchor images and labeled with `0` (indicating dissimilarity).  
* **Combining and Mapping**:  
  * **Concatenate**: Combines positive and negative datasets.  
  * **`preprocess_two(input_image, validation_image, label)`**: Applies preprocessing to both images in each pair and retains the label.

### **4.4 Dataset Preparation (Section 6.4)** {#4.4-dataset-preparation-(section-6.4)}

* **Cache**:  
  * Caches the dataset for faster access.  
* **Shuffle**:  
  * Shuffles the dataset with a buffer size of 10,000 for randomness.  
* **Split**:  
  * **Training Data**:  
    * Takes 70% of the dataset, batches it with a batch size of 16, and prefetches 8 batches.  
  * **Testing Data**:  
    * Takes 30% of the dataset, batches it with a batch size of 16, and prefetches 8 batches.

## **5\. Model Overview** {#5.-model-overview}

This section provides an overview of the model implemented throughout **Section 7**.

### **5.1 Embedding Model (Section 7.1)** {#5.1-embedding-model-(section-7.1)}

* **Purpose**:  
  * **Feature Extraction**: Extracts key features of the image.  
  * **Similarity Comparison**: Generates embeddings that can be compared to determine if two images are similar.  
* **Functionality**:  
  * The embedding model processes an image through a series of convolutional and pooling layers, which gradually distill the image's features.  
  * The final output is a dense vector (embedding) that uniquely represents the input image.  
  * In the context of a Siamese network, these embeddings are used to compare images: similar images will have similar embeddings, while different images will have distinct embeddings.  
* **Use in Facial Verification**:  
  * The model produces embeddings for two images. Similar images have similar embeddings, enabling the Siamese network to verify if the images are of the same person.  
* **Architecture**:  
  * Input layer of shape `(100, 100, 3)` which takes the preprocessed image as the input.  
  * 4 blocks of Convolutional layers with increasing filter sizes (64, 128, 128, 256).  
  * Maxpooling layer after each convolutional layer in each block.  
  * Flatten layer.  
  * Dense layer with 4096 units (sigmoid activation).

### **5.2 L1Dist Layer (Section 7.2)** {#5.2-l1dist-layer-(section-7.2)}

* **Purpose**:  
  * **Similarity Measurement**: Determines how close or different two images are by comparing their embeddings.  
* **Functionality**:  
  * **Input**: Takes two embeddings as input—one from the `input_image` and one from the `validation_image`.  
  * **Operation**: Computes the L1 distance, which is the absolute difference between corresponding elements of the two embeddings.  
  * **Output**: Produces a vector that represents the distance between the two images' features.  
* **Use in Facial Verification**:  
  * A small L1 distance suggests the images are of the same person, while a larger distance indicates they are different.

### **5.3 Siamese Model (Section 7.3)** {#5.3-siamese-model-(section-7.3)}

* **Purpose**:  
  * Evaluates the similarity between two images by comparing their feature embeddings.  
* **Functionality**:  
  * **Inputs**: Takes two images—`input_image` and `validation_image`.  
  * **Process**:  
    * Both images are passed through the same embedding model to generate feature embeddings.  
    * The L1Dist layer computes the absolute difference between the two embeddings.  
    * A final dense layer with sigmoid activation classifies whether the images are of the same person based on the distance.  
* **Use in Facial Verification**:  
  * Produces a similarity score: a score close to `1` indicates the images are likely of the same person, while a score close to `0` indicates they are different.

## **6\. Training the Model** {#6.-training-the-model}

This provides an overview of the model training process detailed in **Section 8**.

### **6.1 Components** {#6.1-components}

* **Loss Function and Optimizer (Section 8.1)**:  
  * **Binary Cross-Entropy**: Used to compute the loss between predicted similarity scores and actual labels.  
  * **Adam Optimizer**: Optimizes the model parameters with a learning rate of `1e-4`.  
* **Checkpoints (Section 8.2)**:  
  * **Path and Prefix**: Checkpoints are saved in the directory `./training_checkpoints` with a file prefix of "ckpt".  
  * **Checkpoint Object**: Saves the optimizer state and the Siamese model's weights.  
* **Gradient Calculation (Section 8.3)**:  
  * **Function**: `gradients(batch)`  
  * **Process**:  
    * Computes the loss using the current batch.  
    * Calculates gradients of the loss with respect to model variables.  
    * Applies gradients to update model weights.  
  * **Returns**: The computed loss value.

### **6.2 Training Loop (Section 8.4)** {#6.2-training-loop-(section-8.4)}

* **Function**: `train(data, EPOCHS)`  
* **Parameters**:  
  * `data`: Training dataset.  
  * `EPOCHS`: Number of epochs for training.  
* **Process**:  
  * Iterates over the specified number of epochs.  
  * Updates a progress bar for each batch.  
  * Prints loss and trains the model by calling the `gradients(batch)` function on each batch.  
  * Calculates the Precision, and Recall metrics for each epoch.  
  * Saves checkpoints every 10 epochs.

## **7\. Evaluating the Model** {#7.-evaluating-the-model}

This section provides an overview of the model evaluation process detailed in **Section 9**.

### **7.1 Workflow** {#7.1-workflow}

* **Initialize Metrics**:  
  * **Recall()**: Captures how many actual positives were correctly identified.  
  * **Precision()**: Reflects the accuracy of positive predictions.  
* **Evaluation Loop**:  
  * For each batch in `test_data`:  
    * **Prediction**: Calculate the similarity score (`yhat`) between `test_input` and `test_validation` using the model.  
    * **Update Metrics**: Update Recall and Precision using the true labels (`ytrue`) and predictions (`yhat`).  
* **Output**:  
  * Print the final Recall and Precision values.

## **8\. Saving the Model** {#8.-saving-the-model}

* **Section 10**: Deals with saving the model in the `models` folder.

## **9\. Image Verification** {#9.-image-verification}

This section provides an overview of how images are verified against a set of verification images, implemented in Section 11\.

### **9.1 Verification Function (Section 11.1)** {#9.1-verification-function-(section-11.1)}

* **Function**: `verification(model, detection_threshold, verification_threshold)`  
* **Purpose**:  
  * To compare a reference image with a set of verification images and determine if they match based on specified detection and verification thresholds.  
* **Process**:  
  * **Load and Preprocess Images**:  
    * Iterates over images in the `verification_images` folder.  
    * Preprocesses both the input image and the current verification image.  
  * **Predict Similarity**:  
    * Uses the model to predict the similarity between the input image and each verification image.  
    * Stores the similarity results.  
  * **Verify**:  
    * Calculates the number of predictions that exceed the `detection_threshold`.  
    * Determines the verification score as the ratio of these positive predictions to the total number of verification images.  
  * **Return**:  
    * The results of the predictions and whether the verification passes based on the `verification_threshold`.  
* **Result**:  
  * `results`: A list of similarity scores.  
  * `verified`: A boolean indicating if the verification passed.

### **9.2 Webcam Capture and Verification (Section 11.2)** {#9.2-webcam-capture-and-verification-(section-11.2)}

* **Process**:  
  * **Capture Frames**:  
    * Continuously captures video frames from the webcam using Haar Cascade classifiers to detect faces. The detected face regions are cropped and resized to 250x250 pixels.  
  * **Save Verification Images**:  
    * Press `v` to save the current frame to the `verification_images` folder.  
    * Capture at least 50 images that are recent and have similar properties, like brightness, as the `input_image` that will be verified later.  
  * **Save Input Image and Verify**:  
    * Press `t` to save the current frame as the `input_image` and run verification against the images in the `verification_images` folder.  
* **Output**:  
  * Displays "You are verified" if the verification passes, otherwise "Unverified".

## **10\. Accessing and Training the Model** {#10.-accessing-and-training-the-model}

To get started with this model:

* **Clone the Repository:** Clone this repository to your local machine using Git.  
* **Run the Jupyter Notebook:** Open the provided Jupyter notebook “Facial Recognition with Siamese Network” in your Jupyter environment and just run it.  
* **Capture Images:**  
  * **Anchors and Positives:** Use the built-in code in Section 5.2 of the notebook to capture images from your webcam (Ensure that images are captured without any facial accessories).  
    * Press **"a"** to save the current frame as an anchor image in the anchor directory. (save around 250\)  
    * Press **"p"** to save the current frame as a positive image in the positives directory (save around 250).  
* **Train the Model:** Follow the instructions in the notebook to preprocess the images, prepare the dataset, and train the model.  
* **Capture Verification Images and Input Image:** After training, use the script in Section 9.2 of the notebook for verification (Ensure that images are captured without any facial accessories):  
  * Press **"v"** to save the current frame as a verification image in the verification\_images folder. Capture at least 50 verification images with similar properties to the input image.  
  * Press **"t"** to save the current frame as the input image and initiate the verification process.
