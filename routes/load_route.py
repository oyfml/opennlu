from flask import render_template, session, redirect, url_for, request
from opennlu import app
import os

RASA_MODEL_FOLDER = 'opennlu/data/model/rasa'
PT_MODEL_FOLDER = 'opennlu/data/model/pytorch'
TF_MODEL_FOLDER = 'opennlu/data/model/tensorflow'

#Change current model (by clicking on slide bar; layout.html)
@app.route('/change')
def change():
    session['model_name'] = request.args['name']
    session['model_type'] = request.args['type']
    return redirect(url_for('info'))

#Display current active model information
@app.route('/info')
def info():
    if 'model_type' not in session:
        return redirect(url_for('model',fail=True))

    if session['model_type'] == 'rasa':
        #Get training data and pipeline info
        data_name = app.config['RASA'].get_model(session['model_name']).data
        pipeline_name = app.config['RASA'].get_model(session['model_name']).pipeline
        if os.path.getsize(data_name) > 0:
            with open(data_name, "r") as f: 
                data = f.read()
        if os.path.getsize(pipeline_name) > 0:
            with open(pipeline_name, "r") as f: 
                pipeline = f.read()
        return render_template('/load/rasa_info.html', data_name=data_name, pipeline_name=pipeline_name, data=data, pipeline=pipeline)
    
    elif session['model_type'] == 'pytorch':
        name = session['model_name']
        config_file = os.path.join(os.path.join(PT_MODEL_FOLDER,name),'config.json')
        if os.path.exists(config_file):
            with open(config_file, "r") as f: 
                data = f.read()
        return render_template('/load/pytorch_info.html', data_name=config_file, data=data)
    
    elif session['model_type'] == 'tensorflow':
        name = session['model_name']
        config_file = os.path.join(os.path.join(TF_MODEL_FOLDER,name),'params.json')
        if os.path.exists(config_file):
            with open(config_file, "r") as f: 
                data = f.read()
        return render_template('/load/tensorflow_info.html', data_name=config_file, data=data)
    
    else:
        return redirect(url_for('model',fail=True))
    

#Load models
@app.route('/load')
def load():
    #Get list of rasa loaded models
    rasa_loaded_list = app.config['RASA'].get_all()
    #Create list of rasa's not loaded models
    rasa_model_list = [model for model in os.listdir(RASA_MODEL_FOLDER) if os.path.isdir(os.path.join(RASA_MODEL_FOLDER, model)) ]
    rasa_not_loaded_list = [model for model in rasa_model_list if not model in rasa_loaded_list]

    #Get list of pytorch loaded models
    pt_loaded_list = app.config['PT'].get_all()
    #Create list of pytorch's not loaded models
    pt_model_list = [model for model in os.listdir(PT_MODEL_FOLDER) if os.path.isdir(os.path.join(PT_MODEL_FOLDER, model)) ]
    pt_not_loaded_list = [model for model in pt_model_list if not model in pt_loaded_list]

    #Get list of tensorflow loaded models
    tf_loaded_list = app.config['TF'].get_all()
    #Create list of tensorflow's not loaded models
    tf_model_list = [model for model in os.listdir(TF_MODEL_FOLDER) if os.path.isdir(os.path.join(TF_MODEL_FOLDER, model)) ]
    tf_not_loaded_list = [model for model in tf_model_list if not model in tf_loaded_list]

    return render_template('/load/index.html',rasa_not_loaded_list=rasa_not_loaded_list,pt_not_loaded_list=pt_not_loaded_list,tf_not_loaded_list=tf_not_loaded_list)

#Add new model to RASA_dict / PT_dict / TF_dict
@app.route('/load/adding')
def adding():
    model_type = request.args['type']
    if model_type == 'rasa':
        #(not loaded) Model selected to be loaded
        session['model_name'] = request.args['model']
        session['model_type'] = model_type
        model_direc = os.path.join(RASA_MODEL_FOLDER,session['model_name'])
        data_path = os.path.join(model_direc,'training_data.json')
        pipeline_path = os.path.join(model_direc,'metadata.json')
        # Load model into rasa dict
        rasa = app.config['RASA'].create_new(session['model_name'],data_path,pipeline_path)
        rasa.load(model_direc)
    elif model_type == 'pytorch':
        session['model_name'] = request.args['model']
        session['model_type'] = model_type
        # Load model into pytorch dict
        pytorch = app.config['PT'].create_new(session['model_name'])
        pytorch.load()
    else:
        session['model_name'] = request.args['model']
        session['model_type'] = model_type
        # Load model into tensorflow dict
        tensorflow = app.config['TF'].create_new(session['model_name'])
        tensorflow.load()
    return redirect(url_for('load'))