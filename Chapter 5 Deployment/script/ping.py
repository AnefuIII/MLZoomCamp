from flask import Flask

app = Flask(__name__)

@app.route('/ping', methods = ['GET'])
def pong():
    return 'OCHE AI'


if '__name__' == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 9696)