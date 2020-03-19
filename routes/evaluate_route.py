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
PLOT_FOLDER = os.path.join('/static', 'plots') # for image openning (relative path)
app.config['PLOT_FOLDER'] = PLOT_FOLDER

PT_CACHE_FOLDER = os.path.join('opennlu/services/pytorch_JointBERT','cached_data')

#Redirect according to current model type
@app.route('/evaluate', methods = ['GET', 'POST'])
def evaluate():
    if 'model_type' in session:
        if session['model_type'] == 'rasa':
            return redirect(url_for('evaluate_rasa'))
        elif session['model_type'] == 'pytorch':
            return redirect(url_for('evaluate_pytorch'))
        else: # session['model_type'] == 'tensorflow'
            return redirect(url_for('evaluate_tensorflow'))
    else: #no models trained yet
        return redirect(url_for('model',fail=True))


#Evaluate & Test interface for rasa model
@app.route('/evaluate/rasa', methods = ['GET', 'POST'])
def evaluate_rasa():
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
        msg_list = ['msg', 'intent_name', 'entity_names', 'intent_score', 'entity_scores', 'entity_types', 'entity_extractors', 'results']
        for key in msg_list:
            if key in session:
                session.pop(key)

        test_list = ['test_data', 'test_name', 'int_metrics', 'ent_metrics', 'int_results', 'ent_results', 'list_size', 'hist_path', 'cf_path', 'intent_report', 'entity_report']
        for key in test_list:
            if key in session:
                session.pop(key)

        #Check if model is loaded
        if app.config['RASA'].size() == 0:
            return redirect(url_for('model',fail=True))
        else:
            return render_template('/evaluate/rasa.html')


#Evaluate & Test interface for pytorch model
@app.route('/evaluate/pytorch', methods = ['GET', 'POST'])
def evaluate_pytorch():
    if request.method == 'POST':
        #Interpret single message
        if 'interpret' in request.form:
            #Get pytorch model
            pytorch = app.config['PT'].get_model(session['model_name'])
            #Perform interpretation
            session['msg'] = request.form['msg']
            [intent, slot, confid_score] = pytorch.predict(session['msg'])
            session['intent_msg'] = intent
            session['slot_msg'] = slot
            session['score'] = str(confid_score)
        else:
            #download multiple files from the folder
            list_folder = request.files.getlist('test_folder') #list()
            #check if folder contains correct files
            file_check = {'label':0, 'seq.in':0, 'seq.out':0}
            for file in list_folder:
                if os.path.basename(file.filename) in file_check:
                    file_check[os.path.basename(file.filename)] = file_check[os.path.basename(file.filename)] + 1           
            if 0 in file_check.values(): #check if filenames meet requirement
                fail = True
                fail_message = 'Files uploaded do not match filename requirements. Please check if your label, text sequence and BIO-tag sequence files are named as label, seq.in and seq.out respectively for system to recognise.'
                return redirect(url_for('evaluate_pytorch',fail=fail,fail_message=fail_message))
            elif not all([False for value in file_check.values() if value>1]): #invalid data folder: contains more than one of each label,seq.in,seq.out files
                fail = True
                fail_message = 'Invalid folder selected! Folder contains more than required number of files (3). Please select the direct parent data folder with only one instance of label, seq.in and seq.out file.'
                return redirect(url_for('evaluate_pytorch',fail=fail,fail_message=fail_message))
            else: 
                # extract data from files
                for file in list_folder:
                    if os.path.basename(file.filename) == 'label':
                        file.seek(0)
                        test_label_name = file.filename
                        test_label_content = file.read().decode("utf-8")
                    elif os.path.basename(file.filename) == 'seq.in':
                        file.seek(0)
                        test_text_name = file.filename
                        test_text_content = file.read().decode("utf-8")
                    elif os.path.basename(file.filename) == 'seq.out':
                        file.seek(0)
                        test_tags_name = file.filename
                        test_tags_content = file.read().decode("utf-8")

                # check if file content satisfy requirements
                len_label = len(test_label_content.splitlines())
                text_ex = test_text_content.splitlines()
                len_text = len(text_ex)
                tags_ex = test_tags_content.splitlines()
                len_tags = len(tags_ex)
                #check if no. of training examples tally
                if ((len_label != len_text) or (len_label != len_tags)): 
                    fail = True
                    fail_message = 'Number of training examples do not match across the 3 files. Please navigate to edit data page for correction.'
                    return redirect(url_for('evaluate_pytorch',fail=fail,fail_message=fail_message))
                #check for each example if token count tally
                for text, tags in zip(text_ex,tags_ex):
                    if len(text.split()) != len(tags.split()):
                        fail = True
                        fail_message = 'Number of word tokens do not match number of tags in BIO-tag sequence. Please navigate to edit data page for correction'
                        return redirect(url_for('evaluate_pytorch',fail=fail,fail_message=fail_message))

                # data safe to save
                test_folder_path = os.path.join(app.config['DATA_FOLDER'],os.path.dirname(test_label_name))
                
                if not os.path.exists(test_folder_path): 
                    os.makedirs(test_folder_path)
                label_file = os.path.join(test_folder_path,'label')
                with open(label_file, "w") as f: 
                    f.write(test_label_content)
                text_file = os.path.join(test_folder_path,'seq.in')
                with open(text_file, "w") as f: 
                    f.write(test_text_content)
                tags_file = os.path.join(test_folder_path,'seq.out')
                with open(tags_file, "w") as f: 
                    f.write(test_tags_content)

                session['test_data'] = test_text_content.splitlines()

                # copy data to cached folder
                from shutil import copyfile
                new_location = os.path.join(PT_CACHE_FOLDER,session['model_name'])

                if not os.path.exists(new_location):
                    os.makedirs(new_location) # create (task) folder
                    os.makedirs(os.path.join(new_location,'test')) # create testing folder
                elif not os.path.exists(os.path.join(new_location,'test')):
                    os.makedirs(os.path.join(new_location,'test')) # create testing folder

                copyfile(os.path.join(test_folder_path,'label'),os.path.join(os.path.join(new_location,'test'),'label'))
                copyfile(os.path.join(test_folder_path,'seq.in'),os.path.join(os.path.join(new_location,'test'),'seq.in'))
                copyfile(os.path.join(test_folder_path,'seq.out'),os.path.join(os.path.join(new_location,'test'),'seq.out'))   

                # Evaluation:
                #get pytorch model
                pytorch = app.config['PT'].get_model(session['model_name'])
                #peform evaluation & get results
                [session['metrics'], session['int_pred'], session['int_true'], session['slot_pred'], session['slot_true'], confid_score] = pytorch.evaluate()
                ### metrics : loss, intent_accuracy, intent_precision, intent_recall, intent_f1, slot_accuracy, slot_precision, slot_recall, slot_f1, sementic_frame_acc
                session['confid_score'] = [str(score) for score in confid_score]
                [intent_report, slot_report] = pytorch.evaluation_get_individual_report()
                session['intent_report'] = json.dumps(intent_report, indent=2)
                session['slot_report'] = json.dumps(slot_report, indent=2)
                session['list_size'] = len(session['int_true']) 
                #plot confusion matrix
                basename = os.path.basename(test_folder_path)
                cf_name = session['model_name'] + '_' + basename + '_cf.png'
                hist_name = session['model_name'] + '_' + basename + '_hist.png'
                session['cf_path'] = os.path.join(app.config['PLOTS_FOLDER'], cf_name)
                session['hist_path'] = os.path.join(app.config['PLOTS_FOLDER'], hist_name)
                if os.path.isfile(session['cf_path']):
                    os.remove(session['cf_path'])
                if os.path.isfile(session['hist_path']): 
                    os.remove(session['hist_path']) 
                pytorch.compute_confusion_matrix(session['cf_path']) #cf and hist stored in static/plots folder
                pytorch.compute_histogram(session['hist_path'])

        #Pre-process: copy data from previous load (if any)
        if 'msg' not in session:
            session['msg'] = ""
        if 'intent_msg' not in session:
            session['intent_msg'] = ""
            session['slot_msg'] = ""
            session['score'] = ""
        
        #Dictionary to pass to view for single msg interpretation
        msg_chunk = {
            'intent_msg':session['intent_msg'],
            'slot_msg':session['slot_msg'],
            'msg':session['msg'],
            'score':session['score']
        }
        #Dictionary to pass to view for testing
        if 'test_data' not in session:
            test_chunk = {
            }
        else:
            test_chunk = {
                'test_data': session['test_data'],
                'metrics': session['metrics'],
                'int_pred': session['int_pred'], 
                'int_true': session['int_true'], 
                'slot_pred': session['slot_pred'], 
                'slot_true': session['slot_true'],
                'list_size': session['list_size'],
                'cf_path': os.path.join(app.config['PLOT_FOLDER'], cf_name),  
                'hist_path': os.path.join(app.config['PLOT_FOLDER'], hist_name), 
                'int_report': session['intent_report'],
                'slot_report': session['slot_report'],
                'confid_score': session['confid_score']
            }
        return render_template('/evaluate/pytorch.html', **msg_chunk, **test_chunk)

    else:
        #Clear session variables (when loading url directly)
        msg_list = ['msg', 'intent_msg', 'slot_msg', 'score']
        for key in msg_list:
            if key in session:
                session.pop(key)

        test_list = ['test_data', 'metrics', 'int_pred', 'int_true', 'slot_pred', 'slot_true', 'list_size', 'cf_path', 'hist_path', 'intent_report', 'slot_report', 'confid_score']
        for key in test_list:
            if key in session:
                session.pop(key)
        
        #Check if model is loaded
        if app.config['PT'].size() == 0:
            return redirect(url_for('model',fail=True))
        else:
            if 'fail' in request.args:
                fail = request.args['fail']
                fail_message = request.args['fail_message']
                return render_template('/evaluate/pytorch.html',fail=fail,fail_message=fail_message)
            else:    
                return render_template('/evaluate/pytorch.html')


#Evaluate & Test interface for tensorflow model
@app.route('/evaluate/tensorflow', methods = ['GET', 'POST'])
def evaluate_tensorflow():
    if request.method == 'POST':
         #Interpret single message
        if 'interpret' in request.form:
            #Get tensorflow model
            tensorflow = app.config['TF'].get_model(session['model_name'])
            #Perform interpretation
            session['msg'] = request.form['msg']
            response = tensorflow.predict(session['msg'])

            session['intent_msg'] = response['intent']['name']
            session['slot_msg'] = response['slots']
            session['score'] = response['intent']['confidence']

        else:
            #download multiple files from the folder
            list_folder = request.files.getlist('test_folder') #list()
            #check if folder contains correct files
            file_check = {'label':0, 'seq.in':0, 'seq.out':0}
            for file in list_folder:
                if os.path.basename(file.filename) in file_check:
                    file_check[os.path.basename(file.filename)] = file_check[os.path.basename(file.filename)] + 1           
            if 0 in file_check.values(): #check if filenames meet requirement
                fail = True
                fail_message = 'Files uploaded do not match filename requirements. Please check if your label, text sequence and BIO-tag sequence files are named as label, seq.in and seq.out respectively for system to recognise.'
                return redirect(url_for('tensorflow',fail=fail,fail_message=fail_message))
            elif not all([False for value in file_check.values() if value>1]): #invalid data folder: contains more than one of each label,seq.in,seq.out files
                fail = True
                fail_message = 'Invalid folder selected! Folder contains more than required number of files (3). Please select the direct parent data folder with only one instance of label, seq.in and seq.out file.'
                return redirect(url_for('tensorflow',fail=fail,fail_message=fail_message))
            else: 
                # extract data from files
                for file in list_folder:
                    if os.path.basename(file.filename) == 'label':
                        file.seek(0)
                        test_label_name = file.filename
                        test_label_content = file.read().decode("utf-8")
                    elif os.path.basename(file.filename) == 'seq.in':
                        file.seek(0)
                        test_text_name = file.filename
                        test_text_content = file.read().decode("utf-8")
                    elif os.path.basename(file.filename) == 'seq.out':
                        file.seek(0)
                        test_tags_name = file.filename
                        test_tags_content = file.read().decode("utf-8")

                # check if file content satisfy requirements
                len_label = len(test_label_content.splitlines())
                text_ex = test_text_content.splitlines()
                len_text = len(text_ex)
                tags_ex = test_tags_content.splitlines()
                len_tags = len(tags_ex)
                #check if no. of training examples tally
                if ((len_label != len_text) or (len_label != len_tags)): 
                    fail = True
                    fail_message = 'Number of training examples do not match across the 3 files. Please navigate to edit data page for correction.'
                    return redirect(url_for('tensorflow',fail=fail,fail_message=fail_message))
                #check for each example if token count tally
                for text, tags in zip(text_ex,tags_ex):
                    if len(text.split()) != len(tags.split()):
                        fail = True
                        fail_message = 'Number of word tokens do not match number of tags in BIO-tag sequence. Please navigate to edit data page for correction'
                        return redirect(url_for('tensorflow',fail=fail,fail_message=fail_message))

                # data safe to save
                test_folder_path = os.path.join(app.config['DATA_FOLDER'],os.path.dirname(test_label_name))
                
                if not os.path.exists(test_folder_path): 
                    os.makedirs(test_folder_path)
                label_file = os.path.join(test_folder_path,'label')
                with open(label_file, "w") as f: 
                    f.write(test_label_content)
                text_file = os.path.join(test_folder_path,'seq.in')
                with open(text_file, "w") as f: 
                    f.write(test_text_content)
                tags_file = os.path.join(test_folder_path,'seq.out')
                with open(tags_file, "w") as f: 
                    f.write(test_tags_content)

                test_data = test_text_content.splitlines()
                session['test_data'] = test_folder_path

                # Evaluation:
                #get tensorflow model
                tensorflow = app.config['TF'].get_model(session['model_name'])
                #peform evaluation & get results
                [metrics, predicted_intents, true_intents, predicted_tags, true_tags, confid_score] = tensorflow.evaluate(test_folder_path)
                ### metrics : intent_accuracy, intent_precision, intent_recall, intent_f1, slot_accuracy, slot_precision, slot_recall, slot_f1
                confid_score = [str(score) for score in confid_score]
                
                [intent_report, slot_report] = tensorflow.evaluation_get_individual_report()
                session['intent_report'] = json.dumps(intent_report, indent=2)
                session['slot_report'] = json.dumps(slot_report, indent=2)
                
                session['list_size'] = len(true_intents) 
                #plot confusion matrix
                basename = os.path.basename(test_folder_path)
                cf_name = session['model_name'] + '_' + basename + '_cf.png'
                hist_name = session['model_name'] + '_' + basename + '_hist.png'
                session['cf_path'] = os.path.join(app.config['PLOTS_FOLDER'], cf_name)
                session['hist_path'] = os.path.join(app.config['PLOTS_FOLDER'], hist_name)
                if os.path.isfile(session['cf_path']):
                    os.remove(session['cf_path'])
                if os.path.isfile(session['hist_path']): 
                    os.remove(session['hist_path']) 
                tensorflow.compute_confusion_matrix(session['cf_path']) #cf and hist stored in static/plots folder
                tensorflow.compute_histogram(session['hist_path'])
                print(hist_name)
                print(session['hist_path'])
                print(os.path.join(app.config['PLOT_FOLDER'], hist_name))

        #Pre-process: copy data from previous load (if any)
        if 'msg' not in session:
            session['msg'] = ""
        if 'intent_msg' not in session:
            session['intent_msg'] = ""
            session['slot_msg'] = ""
            session['score'] = ""
        
        #Dictionary to pass to view for single msg interpretation
        msg_chunk = {
            'intent_msg':session['intent_msg'],
            'slot_msg':session['slot_msg'],
            'msg':session['msg'],
            'score':session['score']
        }
        #Dictionary to pass to view for testing
        if 'test_data' not in session:
            test_chunk = {
            }
        else:
            test_chunk = {
                'test_data': test_data,
                'metrics': metrics,
                'int_pred': predicted_intents, 
                'int_true': true_intents, 
                'slot_pred': predicted_tags, 
                'slot_true': true_tags,
                'list_size': session['list_size'],
                'cf_path': os.path.join(app.config['PLOT_FOLDER'], cf_name),  
                'hist_path': os.path.join(app.config['PLOT_FOLDER'], hist_name), 
                'int_report': session['intent_report'],
                'slot_report': session['slot_report'],
                'confid_score': confid_score
            }
        return render_template('/evaluate/tensorflow.html', **msg_chunk, **test_chunk)
        
    else:
        #Clear session variables (when loading url directly)
        msg_list = ['msg', 'intent_msg', 'slot_msg', 'score']
        for key in msg_list:
            if key in session:
                session.pop(key)

        test_list = ['test_data', 'intent_report', 'slot_report', 'list_size', 'cf_path', 'hist_path']
        for key in test_list:
            if key in session:
                session.pop(key)

        #Check if model is loaded
        if app.config['TF'].size() == 0:
            return redirect(url_for('model',fail=True))
        else:
            if 'fail' in request.args:
                fail = request.args['fail']
                fail_message = request.args['fail_message']
                return render_template('/evaluate/tensorflow.html',fail=fail,fail_message=fail_message)
            else:    
                return render_template('/evaluate/tensorflow.html')

        