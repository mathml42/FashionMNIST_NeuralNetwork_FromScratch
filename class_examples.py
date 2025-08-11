import numpy as np
from keras.datasets import fashion_mnist
import wandb
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # Importing the dataset of fashion MNIST
classes = ["T-shirt/top","Trouser","Pullover", "Dress","Coat"
           ,"Sandal",	"Shirt","Sneaker",	"Bag",	"Ankle boot"] # Name of all classes

wandb.login() 
wandb.init(project="FashionMNIST_NeuralNetwork_FromScratch") # Initialize project with mentioned name if exist, else create a new one with mentioned name

plt.figure(figsize=(6,9)) # fixing figure size
examples = [] # Empty list for showing example classes in wandb
for i in range(10):
    index = np.where(train_labels==i)[0][0]   # finding the first index where i_th label present
    im = plt.subplot(4,3,i+1) # plotting figures in plot using subplot
    im.imshow(train_images[index], cmap="Greys") 
    plt.title(classes[i])   
    plt.axis("off")
    image = wandb.Image(train_images[index], caption=classes[i]) # converting i_th class in the wandb image form
    examples.append(image) 
plt.pause(5)
wandb.log({"Class_examples": examples})