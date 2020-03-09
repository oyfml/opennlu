from flask import render_template, redirect, request, url_for, session
from werkzeug.utils import secure_filename
from opennlu import app
from opennlu.services import rasa_service 
import os
import json


IMAGE_FOLDER = os.path.join('/static', 'img')
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
DATA_FOLDER = os.path.join('opennlu/data', 'data')
app.config['DATA_FOLDER'] = DATA_FOLDER
RASA_FOLDER = os.path.join('opennlu/data', 'rasa_pipeline')
app.config['RASA_FOLDER'] = RASA_FOLDER

#Default model route, select training framework
@app.route('/model')
def model():
    # fail = no model loaded or trained
    if 'fail' in request.args:
        fail = request.args['fail']
    else:
        fail = False
    return render_template('/models/index.html',fail=fail)

#RASA interface
@app.route('/model/rasa', methods = ['GET', 'POST'])
def rasa():
    if request.method == 'POST':
        #Display selected training data file
        if 'data' in request.form:
            data = request.files['datafile']
            data_filename = os.path.join(app.config['DATA_FOLDER'], data.filename)
            data.save(os.path.join(app.config['DATA_FOLDER'],secure_filename(data.filename))) #store in data folder
            if os.path.getsize(data_filename) > 0:
                with open(data_filename, "r") as f: 
                    train_data = f.read()
                    session['data_name'] = data_filename
            if 'config_name' in session:
                with open(session['config_name'], "r") as f: 
                    config_data = f.read()
            else:
                config_data = ""
            return render_template('/models/rasa.html',train_data=train_data, config_data=config_data)
        #Display selected pipeline file
        elif 'config' in request.form:
            config = request.files['ymlfile']
            yml_filename = os.path.join(app.config['RASA_FOLDER'], config.filename)
            config.save(os.path.join(app.config['RASA_FOLDER'],secure_filename(config.filename))) #store in data folder of rasa
            if os.path.getsize(yml_filename) > 0:
                with open(yml_filename, "r") as f: 
                    config_data = f.read()
                    session['config_name'] = yml_filename
            if 'data_name' in session:
                with open(session['data_name'], "r") as f: 
                    train_data = f.read()
            else:
                train_data = ""
            return render_template('/models/rasa.html',train_data=train_data, config_data=config_data)
        #Train RASA
        else:   
            session['model_name'] = request.form['model_id']
            #Both training and config file selected
            if('data_name' in session and 'config_name' in session):
                rasa = app.config['RASA'].create_new(name=session['model_name'],data=session['data_name'],pipeline=session['config_name'])
                rasa.train()
                return redirect(url_for('evaluate'))
            return render_template('/models/rasa.html',ready=False)
    else:
        #clear session variables that pass between routes
        if 'data_name' in session:
            session.pop('data_name')
        if 'config_name' in session:
            session.pop('config_name')
        if 'rasa' in session:
            session.pop('rasa')
        return render_template('/models/rasa.html',ready=False)


#TensorFlow interface
@app.route('/model/tensorflow', methods = ['GET', 'POST'])
def tensorflow():
    
    return render_template('/models/tf.html')

#PyTorch interface
@app.route('/model/pytorch', methods = ['GET', 'POST'])
def pytorch():
    
    return render_template('/models/pt.html')


