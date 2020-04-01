from flask import Flask, session
from opennlu.services.rasa_service import RASA_dict
from opennlu.services.tensorflow_service import TF_dict
from opennlu.services.pytorch_service import PT_dict
from opennlu.services import rasa_service, tensorflow_service, pytorch_service
import tensorflow
import os


class OpenNLU:

    def __init__(self):
        self.flask = Flask(__name__)
    
    def run(self):
        from opennlu import routes
        self.flask.run(debug=True)

# Create Flask Instance
host = OpenNLU() 
app = host.flask   

# Create RASA_dict, tf_dict and PT_dict Instance using application factories
app.config['RASA'] = RASA_dict() 
app.config['TF'] = TF_dict()
app.config['PT'] = PT_dict()

# Initialise secret key for session
app.secret_key = os.urandom(32)


