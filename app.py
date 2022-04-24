import os
import uuid
from flask import Flask,request
from flask_cors import CORS
import base64
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

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
@app.route('/search',methods=['POST','GET'])
def search():
    if request.method == 'POST':
        data = request.get_data()
        data = data.split(b',')[1]
        file = uuid.uuid1()
        filename = f"Datasets/CUFSF/{file}.jpg"
        print(f"filename:{filename}")
        with open(filename,"wb") as img:
            img.write(base64.b64decode(data))
        with os.popen(f"cd  DVG-Face && python eval.py --input {file}") as pipe:
            number = "{:0>5d}".format(int(pipe.read()))
        # pipe = os.popen(f"cd  DVG-Face && python eval.py --input {file}")
        # number = int(pipe.read())
        # pipe.close()
        # release resouces
        os.remove(filename)
        # call functions
        return f"static/lib/{number}.jpg"
    else:
        return "API:search"
@app.route('/sketch',methods=['POST','GET'])
def sketch():
    if request.method == 'POST':
        data = request.get_data()
        data = data.split(b',')[1]
        # filename = f"./Datasets/CUFS-CAGAN/CUHK/sketches/00.png"
        # print(f"filename:{filename}")
        # with open(filename,"wb") as img:
        #     img.write(base64.b64decode(data))
        # os.system("cd CA-GAN && python eval.py --output ../static 00")
        return "static/00.jpg"
    else:
        return "API:sketch"
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5001,debug=True)
