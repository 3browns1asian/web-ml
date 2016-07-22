import os
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

@socketio.on('connect')
def connect_handler(sid):
	print sid + " has connected."

@socketio.on('sendSensorData')
def sensor_data_handler(sid, data):
	print sid + " has sent some data: " + data
	emit('receivedSensorData', data)

if __name__ == '__main__':
	socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))