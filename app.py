import base64
import io
import re
import os
from flask import Flask, jsonify, render_template, request, url_for
from PIL import Image
from scipy.misc import imread, imresize, imsave
import numpy as np
import torch
from model import CNN, model
app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    guess = 0
    if request.method == 'POST':
        # requests image from url
        img_size = 250, 300
        image_url = request.values['imageBase64']
        image_string = re.search(r'base64,(.*)', image_url).group(1)
        image_bytes = io.BytesIO(base64.b64decode(image_string))
        image = Image.open(image_bytes)
        image = image.resize(img_size, Image.LANCZOS)
        image = image.convert('1')
        img = image.convert("RGB")
        file_num = len(os.listdir("./logs/normal/"))
        imsave("check.jpeg", img)
        # imsave("./logs/normal/{0}.jpeg".format(file_num),img)

        guess = model("check.jpeg", file_num)
        f = open("./logs/labels/{}.txt".format(file_num),'w')
        f.write(str(guess))
        f.close()
        try:
            os.remove("check.jpeg")
            os.remove("check1.jpeg")
        except:
            pass
        return jsonify(guess=guess)

    return render_template('index.html', guess=guess)


if __name__ == "__main__":
    app.run()
