from flask import Flask, render_template, request,send_from_directory,Response, request
from commons import process_image, get_model
from inference import predict, view_classify
import numpy as np
import cv2
import os
from PIL import Image
import flask
import io
import json
from werkzeug.utils import secure_filename
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
UPLOAD_FOLDER = './imgdir'
# from inference import get_disease_name
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# from inference import get_flower_name


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('index.html', name="Yuhui")
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('file not uploaded , please try again')
            return
        f = request.files['file']  
        image_name = f.filename
        file_path = os.path.join(
            app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(file_path)
        request.files.get("image")
        image = f.read()
        image = Image.open(file_path)

        model = get_model()
        top_probability, top_class = predict(image, model, topk=5)
        return render_template('result.html', image_name = image_name, top_class=top_class, top_probability=top_probability)

@app.route('/upload/<image_name>')
def send_image(image_name):
    return send_from_directory(UPLOAD_FOLDER,image_name)

@app.route('/show/<image_name>')
def show_image(image_name):
    print("hahahah",image_name)
    fig = Figure()
    image = Image.open(UPLOAD_FOLDER + '/' + image_name)
    model = get_model()
    top_probability, top_class = predict(image, model, topk=5)
    with open('categories.json') as f:
        cat_to_name = json.load(f)
    fig = view_classify(image, top_probability, top_class, cat_to_name)
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")



if __name__ == '__main__':
    app.run(debug=True)
