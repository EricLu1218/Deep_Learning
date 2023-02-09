# Deep Learning
These are "2022 NTHU CS565600 Deep Learning" course projects.

# Competition
## Competition 1: Text Feature Engineering
In this competition, you are provided with a supervised dataset $\rm X$ consisting of the **raw content** of news articles and the **binary popularity** (where $1$ means "popular" and $−1$ means "unpopular", calculated based on the number of shares in online social networking services) of these articles as labels. Your goal is to learn a function $\rm f$ from $\rm X$ that is able to predict the popularity of an unseen news article.

## Competition 2: CNN for Object Detection
In this competition, you have to train a model that recognizes objects in an image. Your goal is to output bounding boxes for objects.

## Competition 3: Reverse Image Caption
In this work, we are interested in translating text in the form of single-sentence human-written descriptions directly into image pixels. For example, "**this flower has petals that are yellow and has a ruffled stamen**" and "**this pink and yellow flower has a beautiful yellow center with many stamens**". You have to develop a novel deep architecture and GAN formulation to effectively translate visual concepts from characters to pixels.

## Competition 4: Recommender Systems
In this competition, you should design a recommender system that recommends movies to users. When a user queries your system with $\rm (UserID, Timestamp)$, your system should **return a list of 10 movies in their MovieIDs** $\rm (MovieID_{1}, MovieID_{2}, ..., MovieID_{10})$ which the user might be interested in.

# Lab Assignment
## Lab 1: Scientific Python 101
This lab guides you through basics of Python for the Deep Learning course and provides some useful references.

## Lab 2: Exploring and Visualizing Data
Here's a generated dataset, with 3 classes and 15 attributes. Your goal is to reduce data dimension to 2 and 3, and then plot 2-D and 3-D visualization on the compressed data, respectively.

## Lab 3: Decision Trees & Random Forests
We try to make predition from another [dataset breast cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) wisconsin. But there are too many features in this dataset. Please try to improve accuracy per feature $\rm \frac{Accuracy}{\#Feature}$.

## Lab 4-1: Perceptron, Adaline, and Optimization
Implement the Adaline with SGD which can set different batch_size ($\rm M$) as parameter. Then, use the [Iris](https://archive.ics.uci.edu/ml/datasets/iris) dataset to fit your model with 3 different $\rm M$ (including $\rm M=1$) and fixed learning rate $\rm \eta$ and print out the accuracy of each model. Last, plot the cost against the number of epochs using different $\rm M$ in one figure.

## Lab 4-2: Linear, Polynomial, and Decision Tree Regression
In this assignment, you need to train regression models on [Beijing PM2.5 dataset](http://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data) in winter of 2014.

1. You have to implement
    * a Linear (Polynomial) regressor
    * a Random Forest regressor
2. You need to show a residual plot for each of your model on both training data and testing data.
3. $\rm R^2$ score need to be larger than 0.72 on testing data.

## Lab 5: Regularization
In this assignment, we would like to predict the success of shots made by basketball players in the NBA.

## Lab 6: Logistic Regression and Evaluation Metrics
Predict the presence or absence of cardiac arrhythmia in a patient.

## Lab 7: KNN, SVM, Data Preprocessing, and Scikit-learn Pipeline
In this assignment, you have to train models and handle quality issues on Mushroom dataset.

## Lab 8: Cross Validation & Ensembling
In this assignment, a dataset called Playground dataset will be used. The goal is to train models using any methods you have learned so far to achieve best accuracy on the testing data. You can plot the train.csv and try to ensemble models that performs well on different competitors.

## Lab 9: TensorFlow 101
A brief introduction to TensorFlow

## Lab 10: Word2Vec
In this assignment, you need to do following things:
1. Devise Word2Vec model by subclassing keras.Model.
2. Train your word2vec model and plot your learning curve.
3. Visualize your embedding matrix by t-SNE.
4. Show top-5 nearest neighbors of two words (pick by yourself).

## Lab 11-1: Convolution Neural Networks
In this assignment, you have to implement the input pipeline of the CNN model and try to write/read tfrecord with the **Oregon Wildlife** dataset.

We provide you with the complete code for the image classification task of the CNN model, but remove the part of the input pipeline. What you need to do is completing this part and training the model for at least 5 epochs.

## Lab 11-2: Visualization & Style Transfer
In this assignment, you need to do following things:

### Part I (A Neural Algorithm of Artistic Style)
1. Implement total variational loss. ```tf.image.total_variation``` is not allowed.
2. Change the weights for the style, content, and total variational loss.
3. Use other layers in the model.
    * You need to calculate both content loss and style loss from different layers in the model
4. Write a brief report. Explain how the results are affected when you change the weights, use different layers for calculating loss.
    * Insert markdown cells in the notebook to write the report.

### Part II (AdaIN)
1. Implement AdaIN layer and use single content image to create 25 images with different styles.

## Lab 12-1: Seq2Seq Learning & Neural Machine Translation
In this assignment, we will train a seq2seq model with **Luong Attention** to solve a sentiment analysis task with the IMDB dataset.

## Lab 12-2: Image Captioning
In this assignment, you have to train a captcha-recognizer which can identify English words in images.

## Lab 13-1: Autoencoder
In this lab, we are going to introduce Autoencoder and Manifold learning.

## Lab 13-2: Generative Adversarial Network (GAN)
In this assignment, you need to do following things:
1. Implement the [Improved WGAN](https://arxiv.org/pdf/1704.00028.pdf).
2. Train the Improved WGAN on [CelebA](https://www.kaggle.com/c/datalab-lab-14-2/data?fbclid=IwAR0z0lDESiGwLJ8o00b2V5YrKq01SpFkx6t2jbeNaWQ7g_MMIllaa1nuYU0#) dataset. Build dataset that **read** and **resize** images to **64 x 64** for training.
3. Show a **gif of generated samples (at least 8 x 8)** to demonstrate the training process and show the **best generated sample(s)**.
4. Draw the **loss curve of discriminator and generator** during training process into **one image**.
5. Write a brief report about what you have done.

## Lab 14: MDP & Q-Learning & SARSA
In this assignment, you need to do following things:
* Change the update rule from Q-learning to SARSA (with the same episodes).
* Give a brief report to discuss the result (compare Q-learning with SARSA based on the game result).

## Lab 15: Deep Reinforcement Learning
In this assignment, you need to do following things:
* Running the code and comprehense it
* Writing your discovery in this notebook
