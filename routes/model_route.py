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
PT_CACHE_FOLDER = os.path.join('opennlu/services/pytorch_JointBERT','cached_data')

#Default model route, select training framework
@app.route('/model')
def model():
    # fail = no model loaded or trained
    if 'fail' in request.args:
        fail = request.args['fail']
    else:
        fail = False
    return render_template('/models/index.html',fail=fail)

#RASA training interface
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
            #Both training and config file selected
            if('data_name' in session and 'config_name' in session):
                session['model_name'] = request.form['model_id']
                session['model_type'] = 'rasa'

                rasa = app.config['RASA'].create_new(name=session['model_name'],data=session['data_name'],pipeline=session['config_name'])
                rasa.train()
                return redirect(url_for('evaluate'))
            return render_template('/models/rasa.html')
    else:
        #clear session variables that pass between routes
        if 'data_name' in session:
            session.pop('data_name')
        if 'config_name' in session:
            session.pop('config_name')
        if 'rasa' in session:
            session.pop('rasa')
        return render_template('/models/rasa.html')


#PyTorch training interface
@app.route('/model/pytorch', methods = ['GET', 'POST'])
def pytorch():
    if request.method == 'POST':
        if 'train_data' in request.form:
            #download multiple files from the folder
            list_folder = request.files.getlist('train_folder') #list()
            #check if folder contains correct files
            file_check = {'label':0, 'seq.in':0, 'seq.out':0}
            for file in list_folder:
                if os.path.basename(file.filename) in file_check:
                    file_check[os.path.basename(file.filename)] = file_check[os.path.basename(file.filename)] + 1           
            if 0 in file_check.values(): #check if filenames meet requirement
                fail = True
                fail_message = 'Files uploaded do not match filename requirements. Please check if your label, text sequence and BIO-tag sequence files are named as label, seq.in and seq.out respectively for system to recognise.'
                return redirect(url_for('pytorch',fail=fail,fail_message=fail_message))
            elif not all([False for value in file_check.values() if value>1]): #invalid data folder: contains more than one of each label,seq.in,seq.out files
                fail = True
                fail_message = 'Invalid folder selected! Folder contains more than required number of files (3). Please select the direct parent data folder with only one instance of label, seq.in and seq.out file.'
                return redirect(url_for('pytorch',fail=fail,fail_message=fail_message))
            else: 
                # extract data from files
                for file in list_folder:
                    if os.path.basename(file.filename) == 'label':
                        file.seek(0)
                        train_label_name = file.filename
                        train_label_content = file.read().decode("utf-8")
                    elif os.path.basename(file.filename) == 'seq.in':
                        file.seek(0)
                        train_text_name = file.filename
                        train_text_content = file.read().decode("utf-8")
                    elif os.path.basename(file.filename) == 'seq.out':
                        file.seek(0)
                        train_tags_name = file.filename
                        train_tags_content = file.read().decode("utf-8")

                # check if file content satisfy requirements
                len_label = len(train_label_content.splitlines())
                text_ex = train_text_content.splitlines()
                len_text = len(text_ex)
                tags_ex = train_tags_content.splitlines()
                len_tags = len(tags_ex)
                #check if no. of training examples tally
                if ((len_label != len_text) or (len_label != len_tags)): 
                    fail = True
                    fail_message = 'Number of training examples do not match across the 3 files. Please navigate to edit data page for correction.'
                    return redirect(url_for('pytorch',fail=fail,fail_message=fail_message))
                #check for each example if token count tally
                for text, tags in zip(text_ex,tags_ex):
                    if len(text.split()) != len(tags.split()):
                        fail = True
                        fail_message = 'Number of word tokens do not match number of tags in BIO-tag sequence. Please navigate to edit data page for correction'
                        return redirect(url_for('pytorch',fail=fail,fail_message=fail_message))

                # data safe to save
                folder_path = os.path.join(app.config['DATA_FOLDER'],os.path.dirname(train_label_name))
                session['train_data'] = folder_path
                if not os.path.exists(folder_path): 
                    os.makedirs(folder_path)
                label_file = os.path.join(folder_path,'label')
                with open(label_file, "w") as f: 
                    f.write(train_label_content)
                text_file = os.path.join(folder_path,'seq.in')
                with open(text_file, "w") as f: 
                    f.write(train_text_content)
                tags_file = os.path.join(folder_path,'seq.out')
                with open(tags_file, "w") as f: 
                    f.write(train_tags_content)

                # update max sequence length
                if 'max_seq_len' in session:
                    curr_len = len(max(open(text_file, 'r'), key=lambda x: len(x.split())).split())
                    prev_len = session['max_seq_len']
                    session['max_seq_len'] = max(curr_len, prev_len)
                else:
                    session['max_seq_len'] = max(len(max(open(text_file, 'r'), key=lambda x: len(x.split())).split()), 50)

                #preserve previous input info
                if 'valid_data' in session:
                    label_file = os.path.join(session['valid_data'],'label')
                    with open(label_file, "r") as f: 
                        valid_label_content = f.read()
                    text_file = os.path.join(session['valid_data'],'seq.in')
                    with open(text_file, "r") as f: 
                        valid_text_content = f.read()
                    tags_file = os.path.join(session['valid_data'],'seq.out')
                    with open(tags_file, "r") as f: 
                        valid_tags_content = f.read()
                else:
                    valid_label_content = ""
                    valid_text_content = ""
                    valid_tags_content = ""
                
                return render_template('/models/pt.html',
                    train_label_content=train_label_content,
                    train_text_content=train_text_content,
                    train_tags_content=train_tags_content,
                    valid_label_content=valid_label_content,
                    valid_text_content=valid_text_content,
                    valid_tags_content=valid_tags_content,
                    max_seq_len=session['max_seq_len']
                )

        elif 'valid_data' in request.form:
            #download multiple files from the folder
            list_folder = request.files.getlist('valid_folder') #list()
            #check if folder contains correct files
            file_check = {'label':0, 'seq.in':0, 'seq.out':0}
            for file in list_folder:
                if os.path.basename(file.filename) in file_check:
                    file_check[os.path.basename(file.filename)] = file_check[os.path.basename(file.filename)] + 1           
            if 0 in file_check.values(): #check if filenames meet requirement
                fail = True
                fail_message = 'Files uploaded do not match filename requirements. Please check if your label, text sequence and BIO-tag sequence files are named as label, seq.in and seq.out respectively for system to recognise.'
                return redirect(url_for('pytorch',fail=fail,fail_message=fail_message))
            elif not all([False for value in file_check.values() if value>1]): #invalid data folder: contains more than one of each label,seq.in,seq.out files
                fail = True
                fail_message = 'Invalid folder selected! Folder contains more than required number of files (3). Please select the direct parent data folder with only one instance of label, seq.in and seq.out file.'
                return redirect(url_for('pytorch',fail=fail,fail_message=fail_message))
            else: 
                # extract data from files
                for file in list_folder:
                    if os.path.basename(file.filename) == 'label':
                        file.seek(0)
                        valid_label_name = file.filename
                        valid_label_content = file.read().decode("utf-8")
                    elif os.path.basename(file.filename) == 'seq.in':
                        file.seek(0)
                        valid_text_name = file.filename
                        valid_text_content = file.read().decode("utf-8")
                    elif os.path.basename(file.filename) == 'seq.out':
                        file.seek(0)
                        valid_tags_name = file.filename
                        valid_tags_content = file.read().decode("utf-8")

                # check if file content satisfy requirements
                len_label = len(valid_label_content.splitlines())
                text_ex = valid_text_content.splitlines()
                len_text = len(text_ex)
                tags_ex = valid_tags_content.splitlines()
                len_tags = len(tags_ex)
                #check if no. of training examples tally
                if ((len_label != len_text) or (len_label != len_tags)): 
                    fail = True
                    fail_message = 'Number of training examples do not match across the 3 files. Please navigate to edit data page for correction.'
                    return redirect(url_for('pytorch',fail=fail,fail_message=fail_message))
                #check for each example if token count tally
                for text, tags in zip(text_ex,tags_ex):
                    if len(text.split()) != len(tags.split()):
                        fail = True
                        fail_message = 'Number of word tokens do not match number of tags in BIO-tag sequence. Please navigate to edit data page for correction'
                        return redirect(url_for('pytorch',fail=fail,fail_message=fail_message))

                # data safe to save
                folder_path = os.path.join(app.config['DATA_FOLDER'],os.path.dirname(valid_label_name))
                session['valid_data'] = folder_path
                if not os.path.exists(folder_path): 
                    os.makedirs(folder_path)
                label_file = os.path.join(folder_path,'label')
                with open(label_file, "w") as f: 
                    f.write(valid_label_content)
                text_file = os.path.join(folder_path,'seq.in')
                with open(text_file, "w") as f: 
                    f.write(valid_text_content)
                tags_file = os.path.join(folder_path,'seq.out')
                with open(tags_file, "w") as f: 
                    f.write(valid_tags_content)

                # update max sequence length
                if 'max_seq_len' in session:
                    curr_len = len(max(open(text_file, 'r'), key=lambda x: len(x.split())).split())
                    prev_len = session['max_seq_len']
                    session['max_seq_len'] = max(curr_len, prev_len)
                else:
                    session['max_seq_len'] = max(len(max(open(text_file, 'r'), key=lambda x: len(x.split())).split()), 50)

                #preserve previous input info
                if 'train_data' in session:
                    label_file = os.path.join(session['train_data'],'label')
                    with open(label_file, "r") as f: 
                        train_label_content = f.read()
                    text_file = os.path.join(session['train_data'],'seq.in')
                    with open(text_file, "r") as f: 
                        train_text_content = f.read()
                    tags_file = os.path.join(session['train_data'],'seq.out')
                    with open(tags_file, "r") as f: 
                        train_tags_content = f.read()
                else:
                    train_label_content = ""
                    train_text_content = ""
                    train_tags_content = ""
                
                return render_template('/models/pt.html',
                    train_label_content=train_label_content,
                    train_text_content=train_text_content,
                    train_tags_content=train_tags_content,
                    valid_label_content=valid_label_content,
                    valid_text_content=valid_text_content,
                    valid_tags_content=valid_tags_content,
                    max_seq_len=session['max_seq_len']
                )

        else:
            # train model    
            if ('train_data' in session and 'valid_data' in session):
                session['model_name'] = request.form['model_id']
                session['model_type'] = 'pytorch'

                # get hyperparameters
                hyperparameters = { #default
                    'model_type': 'bert',
                    'crf': False,
                    'batch_size': 16,
                    'epoch_num': 10,
                    'learn_rate': 5e-5              
                }
                if 'batch_size' in request.form:
                    hyperparameters['batch_size'] = request.form['batch_size']
                    hyperparameters['epoch_num'] = request.form['epoch_num']
                    hyperparameters['learn_rate'] = request.form['learn_rate']
                    hyperparameters['model_type'] = request.form.get('type_sel')
                    if request.form.get('crf_sel') == 'no':
                        hyperparameters['crf'] = False
                    else:
                        hyperparameters['crf'] = True

                # save model datafolder in pytorch cache
                from shutil import copyfile
                new_location = os.path.join(PT_CACHE_FOLDER,session['model_name'])

                if not os.path.exists(new_location):
                    os.makedirs(new_location) # create (task) folder
                    os.makedirs(os.path.join(new_location,'train')) # create training folder
                    os.makedirs(os.path.join(new_location,'dev')) # create validation folder

                train_folder_path = session['train_data']
                copyfile(os.path.join(train_folder_path,'label'),os.path.join(os.path.join(new_location,'train'),'label'))
                copyfile(os.path.join(train_folder_path,'seq.in'),os.path.join(os.path.join(new_location,'train'),'seq.in'))
                copyfile(os.path.join(train_folder_path,'seq.out'),os.path.join(os.path.join(new_location,'train'),'seq.out'))

                valid_folder_path = session['valid_data']
                copyfile(os.path.join(valid_folder_path,'label'),os.path.join(os.path.join(new_location,'dev'),'label'))
                copyfile(os.path.join(valid_folder_path,'seq.in'),os.path.join(os.path.join(new_location,'dev'),'seq.in'))
                copyfile(os.path.join(valid_folder_path,'seq.out'),os.path.join(os.path.join(new_location,'dev'),'seq.out'))

                pytorch = app.config['PT'].create_new(
                    name=session['model_name'],
                    model_type=hyperparameters['model_type'], 
                    batch_size=hyperparameters['batch_size'], 
                    learning_rate=hyperparameters['learn_rate'], 
                    num_train_epochs=hyperparameters['epoch_num'], 
                    max_seq_len=session['max_seq_len'], 
                    use_crf=hyperparameters['crf']
                )
                pytorch.train()
                return redirect(url_for('evaluate'))
            return render_template('/models/pt.html')

    else:
        #clear session variables that pass between routes
        if 'train_data' in session:
            session.pop('train_data')
        if 'valid_data' in session:
            session.pop('valid_data')
        if 'max_seq_len' in session:
            session.pop('max_seq_len')
        #pass fail alert information
        if 'fail' in request.args:
            fail = request.args['fail']
            fail_message = request.args['fail_message']
        else:
            fail = False
            fail_message = ""
        return render_template('/models/pt.html',fail=fail,fail_message=fail_message)


#TensorFlow training interface
@app.route('/model/tensorflow', methods = ['GET', 'POST'])
def tensorflow():
    if request.method == 'POST':
        if 'train_data' in request.form:
            #download multiple files from the folder
            list_folder = request.files.getlist('train_folder') #list()
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
                        train_label_name = file.filename
                        train_label_content = file.read().decode("utf-8")
                    elif os.path.basename(file.filename) == 'seq.in':
                        file.seek(0)
                        train_text_name = file.filename
                        train_text_content = file.read().decode("utf-8")
                    elif os.path.basename(file.filename) == 'seq.out':
                        file.seek(0)
                        train_tags_name = file.filename
                        train_tags_content = file.read().decode("utf-8")

                # check if file content satisfy requirements
                len_label = len(train_label_content.splitlines())
                text_ex = train_text_content.splitlines()
                len_text = len(text_ex)
                tags_ex = train_tags_content.splitlines()
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
                folder_path = os.path.join(app.config['DATA_FOLDER'],os.path.dirname(train_label_name))
                session['train_data'] = folder_path
                if not os.path.exists(folder_path): 
                    os.makedirs(folder_path)
                label_file = os.path.join(folder_path,'label')
                with open(label_file, "w") as f: 
                    f.write(train_label_content)
                text_file = os.path.join(folder_path,'seq.in')
                with open(text_file, "w") as f: 
                    f.write(train_text_content)
                tags_file = os.path.join(folder_path,'seq.out')
                with open(tags_file, "w") as f: 
                    f.write(train_tags_content)

                # update max sequence length
                if 'max_seq_len' in session:
                    curr_len = len(max(open(text_file, 'r'), key=lambda x: len(x.split())).split())
                    prev_len = session['max_seq_len']
                    session['max_seq_len'] = max(curr_len, prev_len)
                else:
                    session['max_seq_len'] = max(len(max(open(text_file, 'r'), key=lambda x: len(x.split())).split()), 50)

                #preserve previous input info
                if 'valid_data' in session:
                    label_file = os.path.join(session['valid_data'],'label')
                    with open(label_file, "r") as f: 
                        valid_label_content = f.read()
                    text_file = os.path.join(session['valid_data'],'seq.in')
                    with open(text_file, "r") as f: 
                        valid_text_content = f.read()
                    tags_file = os.path.join(session['valid_data'],'seq.out')
                    with open(tags_file, "r") as f: 
                        valid_tags_content = f.read()
                else:
                    valid_label_content = ""
                    valid_text_content = ""
                    valid_tags_content = ""
                
                return render_template('/models/tf.html',
                    train_label_content=train_label_content,
                    train_text_content=train_text_content,
                    train_tags_content=train_tags_content,
                    valid_label_content=valid_label_content,
                    valid_text_content=valid_text_content,
                    valid_tags_content=valid_tags_content,
                    max_seq_len=session['max_seq_len']
                )
        elif 'valid_data' in request.form:
            #download multiple files from the folder
            list_folder = request.files.getlist('valid_folder') #list()
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
                        valid_label_name = file.filename
                        valid_label_content = file.read().decode("utf-8")
                    elif os.path.basename(file.filename) == 'seq.in':
                        file.seek(0)
                        valid_text_name = file.filename
                        valid_text_content = file.read().decode("utf-8")
                    elif os.path.basename(file.filename) == 'seq.out':
                        file.seek(0)
                        valid_tags_name = file.filename
                        valid_tags_content = file.read().decode("utf-8")

                # check if file content satisfy requirements
                len_label = len(valid_label_content.splitlines())
                text_ex = valid_text_content.splitlines()
                len_text = len(text_ex)
                tags_ex = valid_tags_content.splitlines()
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
                folder_path = os.path.join(app.config['DATA_FOLDER'],os.path.dirname(valid_label_name))
                session['valid_data'] = folder_path
                if not os.path.exists(folder_path): 
                    os.makedirs(folder_path)
                label_file = os.path.join(folder_path,'label')
                with open(label_file, "w") as f: 
                    f.write(valid_label_content)
                text_file = os.path.join(folder_path,'seq.in')
                with open(text_file, "w") as f: 
                    f.write(valid_text_content)
                tags_file = os.path.join(folder_path,'seq.out')
                with open(tags_file, "w") as f: 
                    f.write(valid_tags_content)

                # update max sequence length
                if 'max_seq_len' in session:
                    curr_len = len(max(open(text_file, 'r'), key=lambda x: len(x.split())).split())
                    prev_len = session['max_seq_len']
                    session['max_seq_len'] = max(curr_len, prev_len)
                else:
                    session['max_seq_len'] = max(len(max(open(text_file, 'r'), key=lambda x: len(x.split())).split()), 50)

                #preserve previous input info
                if 'train_data' in session:
                    label_file = os.path.join(session['train_data'],'label')
                    with open(label_file, "r") as f: 
                        train_label_content = f.read()
                    text_file = os.path.join(session['train_data'],'seq.in')
                    with open(text_file, "r") as f: 
                        train_text_content = f.read()
                    tags_file = os.path.join(session['train_data'],'seq.out')
                    with open(tags_file, "r") as f: 
                        train_tags_content = f.read()
                else:
                    train_label_content = ""
                    train_text_content = ""
                    train_tags_content = ""
                
                return render_template('/models/tf.html',
                    train_label_content=train_label_content,
                    train_text_content=train_text_content,
                    train_tags_content=train_tags_content,
                    valid_label_content=valid_label_content,
                    valid_text_content=valid_text_content,
                    valid_tags_content=valid_tags_content,
                    max_seq_len=session['max_seq_len']
                )

        else:
            # train model    
            if ('train_data' in session and 'valid_data' in session):
                session['model_name'] = request.form['model_id']
                session['model_type'] = 'tensorflow'

                # get hyperparameters
                hyperparameters = { #default
                    'model_type': 'bert',
                    'crf': False,
                    'batch_size': 64,
                    'epoch_num': 5,
                    'learn_rate': 5e-5              
                }
                if 'batch_size' in request.form:
                    hyperparameters['batch_size'] = request.form['batch_size']
                    hyperparameters['epoch_num'] = request.form['epoch_num']
                    hyperparameters['learn_rate'] = request.form['learn_rate']
                    hyperparameters['model_type'] = request.form.get('type_sel')
                    if request.form.get('crf_sel') == 'no':
                        hyperparameters['crf'] = False
                    else:
                        hyperparameters['crf'] = True

                tensorflow = app.config['TF'].create_new(
                    name=session['model_name'],
                    model_type=hyperparameters['model_type'], 
                    batch_size=hyperparameters['batch_size'], 
                    learning_rate=hyperparameters['learn_rate'], 
                    num_train_epochs=hyperparameters['epoch_num'], 
                    use_crf=hyperparameters['crf']
                )
                tensorflow.train(session['train_data'], session['valid_data'])
                return redirect(url_for('evaluate'))
            return render_template('/models/tf.html')

    else:
        #clear session variables that pass between routes
        if 'train_data' in session:
            session.pop('train_data')
        if 'valid_data' in session:
            session.pop('valid_data')
        if 'max_seq_len' in session:
            session.pop('max_seq_len')

        #pass fail alert information
        if 'fail' in request.args:
            fail = request.args['fail']
            fail_message = request.args['fail_message']
        else:
            fail = False
            fail_message = ""
        return render_template('/models/tf.html',fail=fail,fail_message=fail_message)




