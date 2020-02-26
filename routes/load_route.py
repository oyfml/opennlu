from flask import render_template, session, redirect, url_for, request
from opennlu import app
import os

MODEL_FOLDER = 'opennlu/data/model/rasa'

#Change current model
@app.route('/change')
def change():
    session['model_name'] = request.args['name']
    return redirect(url_for('info'))

#Display current active model information
@app.route('/info')
def info():
    #Get training data and pipeline info
    data_name = app.config['RASA'].get_model(session['model_name']).data
    pipeline_name = app.config['RASA'].get_model(session['model_name']).pipeline
    if os.path.getsize(data_name) > 0:
        with open(data_name, "r") as f: 
            data = f.read()
    if os.path.getsize(pipeline_name) > 0:
        with open(pipeline_name, "r") as f: 
            pipeline = f.read()
    return render_template('/load/info.html', data_name=data_name, pipeline_name=pipeline_name, data=data, pipeline=pipeline)

#Load model
@app.route('/load')
def load():
    #Get list of loaded models
    loaded_list = app.config['RASA'].get_all()
    #Create list of not loaded models
    model_list = [model for model in os.listdir(MODEL_FOLDER) if os.path.isdir(os.path.join(MODEL_FOLDER, model)) ]
    not_loaded_list = [model for model in model_list if not model in loaded_list]
    return render_template('/load/index.html',not_loaded_list=not_loaded_list)

#Add new model to RASA_dict
@app.route('/load/adding')
def adding():
    #(not loaded) Model selected to be loaded
    session['model_name'] = request.args['model']
    model_direc = os.path.join(MODEL_FOLDER,session['model_name'])
    data_path = os.path.join(model_direc,'training_data.json')
    pipeline_path = os.path.join(model_direc,'metadata.json')
    # Load model into rasa dict
    rasa = app.config['RASA'].create_new(session['model_name'],data_path,pipeline_path)
    rasa.load(model_direc)
    return redirect(url_for('load'))