from flask import render_template, redirect, request, url_for, session
from werkzeug.utils import secure_filename
from opennlu import app
from opennlu.services import rasa_service 
from rasa.nlu.model import Interpreter
import os
import json


DATA_FOLDER = os.path.join('opennlu/data','data')
app.config['DATA_FOLDER'] = DATA_FOLDER
PLOTS_FOLDER = os.path.join('opennlu/static', 'plots') # for file check
app.config['PLOTS_FOLDER'] = PLOTS_FOLDER
PLOT_FOLDER = os.path.join('/static', 'plots') # for image
app.config['PLOT_FOLDER'] = PLOT_FOLDER

#Evaluate & Test interface
@app.route('/evaluate', methods = ['GET', 'POST'])
def evaluate():
    if request.method == 'POST':
        #Interpret single message
        if 'interpret' in request.form:
            #Get rasa model
            rasa = app.config['RASA'].get_model(session['model_name'])
            #Perform interpretation
            session['msg'] = request.form['msg']
            result = rasa.results(session['msg'])
            session['results'] = json.dumps(result, indent=2)
            session['intent_name'] = result['intent']['name'] #extract info from nested dict
            session['intent_score'] = result['intent']['confidence']
            entities = result['entities']
            if len(entities) != 0:
                session['entity_names'] = [item['value'] for idx, item in enumerate(entities)] #convert dict to list
                session['entity_types'] = [item['entity'] for idx, item in enumerate(entities)]
                session['entity_scores'] = [item['confidence'] for idx, item in enumerate(entities)]
                session['entity_extractors'] = [item['extractor'] for idx, item in enumerate(entities)]
            else: #no entities found
                session['entity_names'] = ['-nil-']
                session['entity_types'] = ['-nil-']
                session['entity_scores'] = ['-']
                session['entity_extractors'] = ['-nil-']
        #Test data set
        else:
            test_data = request.files['testfile']
            test_filename = os.path.join(app.config['DATA_FOLDER'], test_data.filename)
            test_data.save(os.path.join(app.config['DATA_FOLDER'],secure_filename(test_data.filename))) #store in data folder
            with open(test_filename, "r") as f:
                session['test_data'] = f.read()
                session['test_name'] = test_filename
            #Get rasa model
            rasa = app.config['RASA'].get_model(session['model_name'])
            #Peform evaluation & get results
            [session['int_metrics'], session['ent_metrics'], session['int_results'], session['ent_results']] = rasa.evaluate_metrics(test_filename)            
            session['list_size'] = len(session['int_results'][0]) 
            #Plot confusion matrix and histogram
            basename = os.path.basename(session['test_name'])
            cf_name = session['model_name'] + '_' + os.path.splitext(basename)[0] + '_cf.png'
            hist_name = session['model_name'] + '_' + os.path.splitext(basename)[0] + '_hist.png'
            session['cf_path'] = os.path.join(app.config['PLOTS_FOLDER'], cf_name)
            session['hist_path'] = os.path.join(app.config['PLOTS_FOLDER'], hist_name)
            if os.path.isfile(session['cf_path']):
                os.remove(session['cf_path'])
            if os.path.isfile(session['hist_path']): 
                os.remove(session['hist_path']) 
            rasa.compute_confusion_matrix(session['cf_path']) #hist and cf stored in static/plots folder
            rasa.compute_histogram(session['hist_path'])
            [intent_report, entity_report] = rasa.evaluation_get_individual_report()
            session['intent_report'] = json.dumps(intent_report, indent=2)
            session['entity_report'] = json.dumps(entity_report, indent=2)

        #Pre-process: copy data from previous load (if any)
        if 'msg' not in session:
            session['msg'] = ""
        if 'intent_name' not in session:
            session['intent_name'] = ""
            session['intent_score'] = ""
            session['entity_names'] = ""
            session['entity_types'] = ""
            session['entity_scores'] = ""
            session['entity_extractors'] = ""
            session['results'] = ""
        
        #Dictionary to pass to view for single msg interpretation
        msg_chunk = {
            'intent_name':session['intent_name'],
            'intent_score':session['intent_score'],
            'entity_names':session['entity_names'],
            'entity_types':session['entity_types'],
            'entity_scores':session['entity_scores'],
            'entity_extractors':session['entity_extractors'],
            'results':session['results'],
            'msg':session['msg']
        }
        #Dictionary to pass to view for testing
        if 'test_data' not in session:
            test_chunk = {
            }
        else:
            test_chunk = {
                'int_metrics':session['int_metrics'],
                'ent_metrics':session['ent_metrics'],
                'int_results':session['int_results'],
                'ent_results':session['ent_results'],
                'list_size':session['list_size'],
                'hist_path':os.path.join(app.config['PLOT_FOLDER'], hist_name),
                'cf_path':os.path.join(app.config['PLOT_FOLDER'], cf_name),    
                'int_report':session['intent_report'],
                'ent_report':session['entity_report']
            }
        return render_template('/evaluate/rasa.html', **msg_chunk, **test_chunk)
    else:
        #Clear session variables (when loading url directly)
        if 'msg' in session:
            session.pop('msg')
            session.pop('intent_name')
            session.pop('entity_names')
            session.pop('intent_score')
            session.pop('entity_scores')
            session.pop('entity_types')
            session.pop('entity_extractors')
            session.pop('results')
        if 'test_data' in session:
            session.pop('test_data')
            session.pop('test_name')
            session.pop('int_metrics')
            session.pop('ent_metrics')
            session.pop('int_results')
            session.pop('ent_results')
            session.pop('list_size')
            session.pop('hist_path')
            session.pop('cf_path')
            session.pop('intent_report')
            session.pop('entity_report')

        #Check if model is loaded
        if app.config['RASA'].size() > 0:
            return render_template('/evaluate/rasa.html')
        else:
            return redirect(url_for('model',fail=True))
        