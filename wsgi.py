import sys
import os

path = os.path.expanduser('~/exam_monitoring')
if path not in sys.path:
    sys.path.append(path)

from server import app, socketio

application.secret_key = 'exam_secret'

if __name__ == "__main__":
    socketio.run(app)
