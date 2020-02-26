from flask import render_template, redirect, request, url_for
from werkzeug.utils import secure_filename
from opennlu import app
import os

UPLOAD_FOLDER = 'opennlu/data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

#Default edit route, select current data file from directory or create new data file
@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        #Create new file
        if 'create' in request.form:
            new = request.form['new_name']
            return redirect(url_for('content', name=new, content=""))
        #Open existing file
        elif 'open' in request.form:
            #Save at local directory
            curr = request.files['file']
            curr.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(curr.filename)))
            return redirect(url_for('content', name=secure_filename(curr.filename))) 
        #Convert file type
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
            file1 = "opennlu/data/" + secure_filename(curr_1.filename)
            file2 = "opennlu/data/" + secure_filename(curr_2.filename)
            #Perform Merge
            td = training_data.load_data(file1)
            td = td.merge(load_data(file2))
            content = td.nlu_as_markdown() # always save as markdown
            new_name = request.form['merge_name']
            return redirect(url_for('content', name=new_name, content=content, merge=True))
        #Train-test split
        else:
            from rasa.nlu import training_data

            curr = request.files['train_test_split']
            fraction = int(request.form['rangeSlider'])/100
            file = "opennlu/data/" + secure_filename(curr.filename)
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
    file = "opennlu/data/" + name
    if os.path.exists(file):
        #Open & read content of existing file
        with open(file, "r") as f: 
            title = os.path.splitext(os.path.basename(f.name))[0] #filename w/o extension
            content = f.read()
        extension = os.path.splitext(os.path.basename(f.name))[1] #extension
    else:
        #Default text template for new file
        title = name #filename w/o extension
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
    file = "opennlu/data/" + name
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
        file = "opennlu/data/" + name
        new_content = request.form['newcontent']
        with open(file, "w", newline="\n") as f:
            f.write(new_content) 
        return redirect(url_for('edit'))
    else:
        from rasa.nlu import training_data

        name = request.args.get('name')
        file = "opennlu/data/" + name
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
        trainfile = "opennlu/data/" + trainname
        train_content = request.form['train_content']
        with open(trainfile, "w", newline="\n") as f:
            f.write(train_content) 
        testfile = "opennlu/data/" + testname
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
    