import json
from commons import process_image, get_model,imshow
from PIL import Image
from torch.autograd import Variable
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from flask import Flask, Response, request
import io
from matplotlib.figure import Figure

with open('categories.json') as f:
	cat_to_name = json.load(f)

with open('class_to_idx.json') as f:
	class_to_idx = json.load(f)

idx_to_class = {v:k for k, v in class_to_idx.items()}


def predict(image, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    
    image = torch.FloatTensor([process_image(image)])
    model.eval()
    output = model.forward(Variable(image))
    pobabilities = torch.exp(output).data.numpy()[0]
    

    top_idx = np.argsort(pobabilities)[-topk:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]
    print("top possibilty and classes")
    print(top_class,top_probability)

    return top_probability, top_class



# Display an image along with the top 5 classes


def view_classify(img, probabilities, classes, mapper):
    ''' Function for viewing an image and it's predicted classes.
    '''
    fig = Figure()
    ax2 = fig.add_subplot(2,1,1)

    
    y_pos = np.arange(len(probabilities))
    ax2.barh(y_pos, probabilities)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([mapper[x] for x in classes])
    ax2.invert_yaxis()
    return fig