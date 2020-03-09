from flask import render_template, redirect, request, url_for, session
from werkzeug.utils import secure_filename
from opennlu import app
import os

UPLOAD_FOLDER = 'opennlu/data/data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

############################### RASA ######################################

#Default edit route (Rasa data)
@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        #Create new file
        if 'create' in request.form:
            new = request.form['new_name'] + '.md'
            return redirect(url_for('content', name=new, content=""))
        #Open existing file
        elif 'open' in request.form:
            #Save at local directory
            curr = request.files['file']
            curr.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(curr.filename)))
            return redirect(url_for('content', name=secure_filename(curr.filename))) 
        #Convert file type (md <-> json)
        elif 'convert' in request.form:
            #Save at local directory
            curr = request.files['convert_file']
            curr.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(curr.filename)))
            return redirect(url_for('convert', name=secure_filename(curr.filename))) 
        #Merge training data
        elif 'merge' in request.form:
            from rasa.nlu import training_data, load_data
            #Save at local directory
            curr_1 = request.files['merge_file_1']
            curr_1.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(curr_1.filename)))
            curr_2 = request.files['merge_file_2']
            curr_2.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(curr_2.filename)))
            file1 = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(curr_1.filename))
            #app.config['UPLOAD_FOLDER'] + secure_filename(curr_1.filename)
            file2 = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(curr_2.filename))
            #app.config['UPLOAD_FOLDER'] + secure_filename(curr_2.filename)
            #Perform Merge
            td = training_data.load_data(file1)
            td = td.merge(load_data(file2))
            content = td.nlu_as_markdown() # always save as markdown
            new_name = request.form['merge_name']
            return redirect(url_for('content', name=new_name, content=content, merge=True))
        #Train-test split
        else:
            from rasa.nlu import training_data, load_data

            curr = request.files['train_test_split']
            curr.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(curr.filename)))
            fraction = int(request.form['rangeSlider'])/100
            file = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(curr.filename))
            #app.config['UPLOAD_FOLDER'] + secure_filename(curr.filename)
            td = training_data.load_data(file)
            #Peform Train Test Split
            data_train, data_test = td.train_test_split(train_frac=fraction)
            train_name = os.path.splitext(os.path.basename(curr.filename))[0] + "_train"
            test_name = os.path.splitext(os.path.basename(curr.filename))[0] + "_test"
            data_train = data_train.nlu_as_markdown()
            data_test = data_test.nlu_as_markdown()
            return redirect(url_for('split', train_name=train_name, test_name=test_name, data_train=data_train, data_test=data_test))
    else:
        return render_template('/edit/index.html') 

#Read & display contents in existing or new data file; takes in filename
@app.route('/edit/file', methods=['GET', 'POST'])
def content():
    name = request.args.get('name')
    file = os.path.join(app.config['UPLOAD_FOLDER'],name)
    if os.path.exists(file):
        #Open & read content of existing file
        with open(file, "r") as f: 
            title = os.path.splitext(os.path.basename(f.name))[0] #filename w/o extension
            content = f.read()
        extension = os.path.splitext(os.path.basename(f.name))[1] #extension
    else:
        #Default text template for new file
        title = os.path.splitext(os.path.basename(name))[0] #filename w/o extension
        content = request.args.get('content')
        extension = '.md' #default

    #Flag for editor to fix .md extension for merge feature
    if request.args.get('merge') != None:
        merge = True
    else:
        merge = False

    if request.method == 'POST':
        return redirect(url_for('save'))
    else:
        return render_template('/edit/editor.html', filename=title, filecontent=content, fileext=extension, merge=merge)

#Write changes made in editor to data file
@app.route('/edit/save', methods=['GET', 'POST'])
def save():
    name = request.form['title']
    if 'type' in request.form:
        extension = request.form['type']
    else:
        extension = '.md'
    if extension == '.md':
        name = name + '.md'
    else:
        name = name + '.json'
    file = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(name))
    #Save any changes written in textbox to file
    new_content = request.form['content']
    with open(file, "w", newline="\n") as f:
        f.write(new_content)            
    return redirect(url_for('edit'))

#Convert between markdown and json
@app.route('/edit/convert', methods=['GET', 'POST'])
def convert():
    if request.method == 'POST':
        name = request.form['new_title']
        file = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(name))
        new_content = request.form['newcontent']
        with open(file, "w", newline="\n") as f:
            f.write(new_content) 
        return redirect(url_for('edit'))
    else:
        from rasa.nlu import training_data

        name = request.args.get('name')
        file = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(name))
        #Open & read content of exisiting file
        with open(file, "r") as f: 
            title = os.path.basename(f.name) #filename
            content = f.read()
        extension = os.path.splitext(os.path.basename(f.name))[1] #extension
        if extension == ".md":
            td = training_data.load_data(file)
            output = td.nlu_as_json(indent=2)
            extension = ".json"
        else:
            td = training_data.load_data(file)
            output = td.nlu_as_markdown()
            extension = ".md"
        newfilename =  os.path.splitext(os.path.basename(f.name))[0] + extension
        return render_template('/edit/convert.html',filename=title,newfilename=newfilename,oldcontent=content,newcontent=output)

#Split dataset into train and test set
@app.route('/edit/split', methods=['GET', 'POST'])
def split():
    if request.method == 'POST':
        #Save train and test files
        trainname = request.form['train_title'] + ".md"
        testname = request.form['test_title'] + ".md"
        trainfile = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(trainname))
        train_content = request.form['train_content']
        with open(trainfile, "w", newline="\n") as f:
            f.write(train_content) 
        testfile = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(testname))
        test_content = request.form['test_content']
        with open(testfile, "w", newline="\n") as f:
            f.write(test_content) 
        return redirect(url_for('edit'))
    else:
        #Pass value from prev route
        trainname = request.args.get('train_name')
        testname = request.args.get('test_name')
        traincontent = request.args.get('data_train') 
        testcontent = request.args.get('data_test')
        return render_template('/edit/split.html',trainname=trainname,testname=testname,train_content=traincontent,test_content=testcontent)

###############################  TF/PT  ###############################

# Editor for Tensorflow/PyTorch data
@app.route('/edit/tf-pt', methods=['GET', 'POST'])
def edit_tf_pt():
    if request.method == 'POST':
        if 'create' in request.form: #create new data folder
            folder_name = request.form['new_name']
            folder_path = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(folder_name))
            label_path = os.path.join(folder_path,'label')
            text_path = os.path.join(folder_path,'seq.in')
            tags_path = os.path.join(folder_path,'seq.out')
            if not os.path.exists(folder_path): #create new folder & files if dont exist
                os.makedirs(folder_path)
                os.mknod(label_path)
                os.mknod(text_path)
                os.mknod(tags_path)
            else: #create files in folder if dont exist
                if not os.path.exists(label_path):
                    os.mknod(label_path)
                if not os.path.exists(text_path):
                    os.mknod(text_path)
                if not os.path.exists(tags_path):
                    os.mknod(tags_path)
            return redirect(url_for('content_tf_pt',path=folder_path))
        
        elif 'open' in request.form: #edit existing data folder
            #download multiple files from the folder
            list_folder = request.files.getlist('folder') #list()
            #check if folder contains correct files
            file_check = {'label':0, 'seq.in':0, 'seq.out':0}
            for file in list_folder:
                if os.path.basename(file.filename) in file_check:
                    file_check[os.path.basename(file.filename)] = file_check[os.path.basename(file.filename)] + 1           
            if 0 in file_check.values(): #check if filenames meet requirement
                fail = True
                fail_message = 'Files uploaded do not match filename requirements. Please check if your label, text sequence and BIO-tag sequence files are named as label, seq.in and seq.out respectively for system to recognise.'
                return redirect(url_for('edit_tf_pt',fail=fail,fail_message=fail_message))
            elif not all([False for value in file_check.values() if value>1]): #invalid data folder: contains more than one of each label,seq.in,seq.out files
                fail = True
                fail_message = 'Invalid folder selected! Folder contains more than required number of files (3). Please select the direct parent data folder with only one instance of label, seq.in and seq.out file.'
                return redirect(url_for('edit_tf_pt',fail=fail,fail_message=fail_message))
            else: #success
                for file in list_folder:
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename)) #save files into folder
                folder_path = os.path.join(app.config['UPLOAD_FOLDER'],os.path.dirname(list_folder[0].filename))
                return redirect(url_for('content_tf_pt',path=folder_path))
        
        elif 'convert_rasa' in request.form: #convert rasa data file to tf/pt format
            from rasa.nlu import training_data, load_data
            from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

            curr = request.files['convert_rasa_file']
            curr.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(curr.filename)))
            file = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(curr.filename))

            td = training_data.load_data(file)
            formatted_examples = [ example.as_dict_nlu() for example in td.training_examples ]
            labels = [ex['intent'] for ex in formatted_examples]

            #Tokenize and clean text
            white_space_tokenizer = WhitespaceTokenizer()
            sentences = list()
            BIO_tagging = list()
            types = dict()
            for ex in formatted_examples:
                #Tokenize by white space
                white_space_tokens = white_space_tokenizer.tokenize(ex['text'])
                tokens = [token.text for token in white_space_tokens]
                #Form into input sentence
                sentence = ' '.join(tokens)
                sentences.append(sentence) #seq.in
                #Perform entity tagging
                if 'entities' in ex: #entity exists
                    ent_values = [entity['value'] for entity in ex['entities']] #entity value
                    ent_length = [len(value.split()) for value in ent_values] #length of entity word
                    ent_types = [entity['entity'] for entity in ex['entities']] #entity type
                    #form BI tags
                    for idx, typ in enumerate(ent_types):
                        ent_types[idx] = 'B-' + typ + ''.join([' I-' + typ]*(ent_length[idx] - 1))
                        types['B-' + typ] = True
                        types['I-' + typ] = True
                        #replace sentence with BI
                        sentence = sentence.replace(ent_values[idx].strip(),ent_types[idx].strip()) #and, remove leading and trailing spaces
                    tag_seq = sentence.split()
                    for idx, token in enumerate(tag_seq):
                        #replace sentence with O
                        if token not in types:
                            tag_seq[idx] = 'O'
                #no entity
                else: 
                    tag_seq = ['O' for t in tokens]
                tags = ' '.join(tag_seq)
                BIO_tagging.append(tags)
            
            file_chunk = {
                'folder_name':os.path.splitext(os.path.basename(file))[0],
                'label_name':'label',
                'text_name':'seq.in',
                'tags_name':'seq.out',
                'label_content':'\n'.join([str(i) for i in labels]) + '\n',
                'text_content':'\n'.join([str(i) for i in sentences]) + '\n',
                'tags_content':'\n'.join([str(i) for i in BIO_tagging]) + '\n'
            }
            return render_template('/edit/editor_3.html', **file_chunk) 
        
        else: #convert tf/pt data file to rasa format
            #download multiple files from the folder
            list_folder = request.files.getlist('convert_tf_pt_folder') #list()
            #check if folder contains correct files
            file_check = {'label':0, 'seq.in':0, 'seq.out':0}
            for file in list_folder:
                if os.path.basename(file.filename) in file_check:
                    file_check[os.path.basename(file.filename)] = file_check[os.path.basename(file.filename)] + 1           
            if 0 in file_check.values(): #check if filenames meet requirement
                fail = True
                fail_message = 'Files uploaded do not match filename requirements. Please check if your label, text sequence and BIO-tag sequence files are named as label, seq.in and seq.out respectively for system to recognise.'
                return redirect(url_for('edit_tf_pt',fail=fail,fail_message=fail_message))
            elif not all([False for value in file_check.values() if value>1]): #invalid data folder: contains more than one of each label,seq.in,seq.out files
                fail = True
                fail_message = 'Invalid folder selected! Folder contains more than required number of files (3). Please select the direct parent data folder with only one instance of label, seq.in and seq.out file.'
                return redirect(url_for('edit_tf_pt',fail=fail,fail_message=fail_message))
            else: #success
                for file in list_folder:
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename)) #save files into folder
                folder_path = os.path.join(app.config['UPLOAD_FOLDER'],os.path.dirname(list_folder[0].filename))
                return redirect(url_for('content_to_rasa',path=folder_path))
    
    else:
        if 'fail' in request.args:
            fail = request.args.get('fail')
            fail_msg = request.args.get('fail_message')
        else:
            fail = False
            fail_msg = ""
        return render_template('/edit/index_tf-pt.html',fail=fail,fail_message=fail_msg)

# Edit route for Tensorflow/PyTorch data
@app.route('/edit/tf-pt/file', methods=['GET', 'POST'])
def content_tf_pt():
    if request.method == 'POST':
        return redirect(url_for('save_tf_pt'))
    else:     
        folder_path = request.args.get('path')
        label_path = os.path.join(folder_path,'label')
        text_path = os.path.join(folder_path,'seq.in')
        tags_path = os.path.join(folder_path,'seq.out')
        label_data = ""
        text_data = ""
        tags_data = ""
        if os.path.exists(label_path):
            with open(label_path, "r") as f: 
                label_data = f.read()
        if os.path.exists(text_path):
            with open(text_path, "r") as f: 
                text_data = f.read()
        if os.path.exists(tags_path):
            with open(tags_path, "r") as f: 
                tags_data = f.read()

        file_chunk = {
            'folder_name':os.path.splitext(os.path.basename(folder_path))[0],
            'label_name':'label',
            'text_name':'seq.in',
            'tags_name':'seq.out',
            'label_content':label_data,
            'text_content':text_data,
            'tags_content':tags_data
        }
        return render_template('/edit/editor_3.html', **file_chunk) 

#Write changes made in editor to data folder (3 files)
@app.route('/edit/save/tf-pt', methods=['GET', 'POST'])
def save_tf_pt():
    #Save any changes written in textbox to file
    folder_name = request.form['folder_title']
    label_name = 'label' 
    text_name = 'seq.in' 
    tags_name = 'seq.out'
    label_content = request.form['label_content']
    text_content = request.form['text_content']
    tags_content = request.form['tags_content']
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'],folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file = os.path.join(folder_path,secure_filename(label_name))
    with open(file, "w", newline="\n") as f:
        f.write(label_content)    
    file = os.path.join(folder_path,secure_filename(text_name))
    with open(file, "w", newline="\n") as f:
        f.write(text_content) 
    file = os.path.join(folder_path,secure_filename(tags_name))
    with open(file, "w", newline="\n") as f:
        f.write(tags_content)      
    return redirect(url_for('edit_tf_pt'))

# Convert tensorflow/pytorch format to rasa
@app.route('/edit/tf-pt/rasa', methods=['GET', 'POST'])
def content_to_rasa():
    if request.method == 'POST':
        a = 1
    else:
        folder_path = request.args.get('path')
        label_path = os.path.join(folder_path,'label')
        text_path = os.path.join(folder_path,'seq.in')
        tags_path = os.path.join(folder_path,'seq.out')
        with open(label_path, "r") as f: 
            label_data = f.read()
        with open(text_path, "r") as f: 
            text_data = f.read()
        with open(tags_path, "r") as f: 
            tags_data = f.read()

        #clean data
        label_data = label_data.split()
        label_data = list(filter(None, label_data))  #remove empty strings
        text_data = text_data.split("\n")
        text_data = [text.split() for text in text_data]
        tags_data = tags_data.split("\n")
        tags_data = [tags.split() for tags in tags_data]
        tags_data = list(filter(None, tags_data)) #remove empty sub arrays
        text_data = list(filter(None, text_data)) #remove empty sub arrays

        #record entity count and start index position of entities
        ent_count = [sum('B-' in token for token in ex) for ex in tags_data] #number of entities
        B_idx = [[i for i,token in enumerate(ex) if 'B-' in token] for ex in tags_data] #starting index of entity

        #get end index position of entities & convert from BIO -> rasa format
        for idx,ex in enumerate(text_data):
            for i in range(int(ent_count[idx])):
                start_idx = B_idx[idx][i]
                curr_idx = start_idx + 1 #look at next token
                while(curr_idx < len(tags_data[idx]) and tags_data[idx][curr_idx].startswith('I-')):
                    curr_idx = curr_idx + 1
                end_idx = (curr_idx - 1) #end index of entity
                #add rasa format for entities   
                ex[start_idx] = "[" + ex[start_idx]
                ex[end_idx] = ex[end_idx] + "]" + "(" + tags_data[idx][start_idx][2:] + ")"
        rasa_format = list()
        for i,ex in enumerate(text_data):
            rasa_format.append('- ' + ' '.join(ex))

        #group examples by intent
        intent_dict = dict()
        for i,ex in enumerate(rasa_format):
            if label_data[i] not in intent_dict:
                intent_dict[label_data[i]] = [ex]
            else:
                intent_dict[label_data[i]].append(ex)

        output = ""
        for key in intent_dict.keys():
            output = output + "## intent:" + key + "\n"
            examples = '\n'.join(intent_dict[key])
            output = output + examples + "\n\n\n"

        return render_template('/edit/editor_to_rasa.html',filename=os.path.basename(folder_path),filecontent=output) 


#Write changes made in editor to rasa file
@app.route('/edit/tf-pt/rasa/save', methods=['GET', 'POST'])
def save_to_rasa():
    #Save any changes written in textbox to file
    file_name = request.form['title'] + ".md"
    content = request.form['content']
    file = os.path.join(app.config['UPLOAD_FOLDER'],file_name)
    with open(file, "w") as f:
        f.write(content)          
    return redirect(url_for('edit_tf_pt'))


