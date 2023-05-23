import io
import os
import sys
import uuid
import time
from PIL import Image
from flask import Flask, request
from flask_cors import cross_origin, CORS
import base64

# dirname = os.path.dirname(__file__)
# sys.path.append(os.path.join(dirname, '..'))
sys.path.append('./MyGanNet/')

import torch
import torchvision
from torchvision import transforms
from MyGanNet.models.MyGanModel import MyGanModel
from MyGanNet.options.eval_options import EvalOptions


app = Flask(__name__)
cors = CORS(app, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

opt = EvalOptions().parse()
opt.architecture = 'SE'

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.6022, 0.4003),
])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gan = MyGanModel(opt, False, device)
gan.load_models('./static/models/SE_global_350/')
gan.set_models_train(False)


def clear_dir(dir):
    for f in os.listdir(dir):
        print(os.path.join(dir, f))
        os.remove(os.path.join(dir, f))

# @cross_origin()
@app.route('/')
def hello():
    """
    @api [GET] /
    @apiVersion 1.0.0
    @apiName test
    @apiGroup Test
    @apiDescription 测试API 测试服务器是否在线

    @apiSuccessExample Response-Success:
        'Hello World!'
    @apiErrorExample Response-Fail:
        no response
    """
    return 'Hello World!'


# @cross_origin()
@app.route('/search', methods=['POST', 'GET'])
def search():
    if request.method == 'POST':
        data = request.get_data()
        data = data.split(b',')[1]
        file = uuid.uuid1()
        # filename = f"Datasets/CUFSF/{file}.jpg"
        filename = f"./search_temp/{file}.jpg"
        print(f"filename:{filename}")
        with open(filename, "wb") as img:
            img.write(base64.b64decode(data))
        with os.popen(f"cd  ../DVG-Face && python eval.py --input {file}") as pipe:
            number = "{:0>5d}".format(int(pipe.read()))
        # pipe = os.popen(f"cd  DVG-Face && python eval.py --input {file}")
        # number = int(pipe.read())
        # pipe.close()
        # release resouces
        os.remove(filename)
        # call functions
        # return f"static/lib/{number}.jpg"
        return f"static/lib/{number}.jpg"
    else:
        return "API:search"


# @cross_origin()
@app.route('/sketch', methods=['POST', 'GET'])
def sketch():
    if request.method == 'POST':
        data = request.get_data()
        name = data.split(b',')[2]
        print(name, type(name))
        name = name.decode('utf-8').strip('.png').strip('.jpg')
        print(name, type(name))
        data = data.split(b',')[1]
        print(name)
        filename = f"./sketch_temp/sketch/00.png"
        print(f"filename:{filename}")
        with open(filename, "wb") as img:
            img.write(base64.b64decode(data))
        # os.system("cd ../face-parsing.PyTorch && python test_save_mat.py")
        print("Begin generating!")
        os.system(f"cd ../CA-GAN && python eval.py --output ../static {name}")
        return f"static/{name}.png"
    else:
        return "API:sketch"


# @cross_origin()
@app.route('/generate', methods=['POST', 'GET'])
def generate():
    if request.method == 'POST':
        data = request.get_data()
        _, img_data, _ = data.split(b',')
        
        img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert('L')
        img = trans(img).unsqueeze(0).to(device)
        out = gan.do_forward(img)
        
        clear_dir('static/imgs/')
        torchvision.utils.save_image(out, f'static/imgs/{int(time.time())}.png')
        return f"static/imgs/{int(time.time())}.png"
    else:
        
        return "API:generate"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
