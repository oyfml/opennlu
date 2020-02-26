from flask import render_template, session, redirect, url_for, request
from opennlu import app
import os

#Home route
@app.route('/')
@app.route('/home')
def index():
    return render_template('/home/index.html')

#Global variables; always run before any app.route load
@app.context_processor
def context_processor():
    rasa_icon = os.path.join(app.config['IMAGE_FOLDER'], 'rasa.png')
    tf_icon = os.path.join(app.config['IMAGE_FOLDER'], 'tf.png')
    pt_icon = os.path.join(app.config['IMAGE_FOLDER'], 'pt.png')

    if 'model_name' in session:
        if app.config['RASA'].size() == 0:
            session.pop('model_name')
            curr_model = ""
        else:
            curr_model = session['model_name']
    else:
        curr_model = ""

    return dict(
        rasa_icon=rasa_icon, 
        rasa_model="RASA NLU",
        tf_icon=tf_icon,
        tf_model="TensorFlow",
        pt_icon=pt_icon,
        pt_model="PyTorch",
        curr_model=curr_model,
        rasa_list=app.config['RASA'].get_all(),
        rasa_list_size=app.config['RASA'].size()
    )