from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

@socketio.on('connect')
def connect_handler():
	print "user connected"
	socketio.emit('you\'re connected')

if __name__ == '__main__':
	port = int(os.environ.get("PORT", 5000))
	socketio.run(app, '127.0.0.1', port)