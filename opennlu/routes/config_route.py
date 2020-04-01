from flask import render_template, redirect, request, url_for
from werkzeug.utils import secure_filename
from opennlu import app
import os

RASA_FOLDER = os.path.join('opennlu/data', 'rasa_pipeline')
app.config['RASA_FOLDER'] = RASA_FOLDER

#Rasa pipeline route, select current config file from directory or create new config file
@app.route('/config', methods=['GET', 'POST'])
def config():
    if request.method == 'POST':
        #Create new file
        if 'create' in request.form:
            new = request.form['new_name']
            return redirect(url_for('content_config', name=secure_filename(new+".yml")))
        #Open existing file
        else:
            curr = request.files['file']
            curr.save(os.path.join(app.config['RASA_FOLDER'],secure_filename(curr.filename)))
            return redirect(url_for('content_config', name=secure_filename(curr.filename))) 

    else:
        return render_template('/config/index.html') 

#Read & display contents in existing or new data file; takes in filename
@app.route('/config/file', methods=['GET', 'POST'])
def content_config():
    name = request.args.get('name')
    file = app.config['RASA_FOLDER']+"/" +name
    if os.path.exists(file):
        #Open & read content of exisiting file
        with open(file, "r") as f: 
            title = os.path.basename(f.name)
            content = f.read()
    else:
        #Default text template for new file
        title = name
        content = "pipeline:"
    if request.method == 'POST':
        return redirect(url_for('save_config'))
    else:
        return render_template('/config/editor.html', filename=title, filecontent=content)

#Write changes made in editor to data file
@app.route('/config/save', methods=['GET', 'POST'])
def save_config():
    name = request.form['title']
    file = app.config['RASA_FOLDER']+"/" +name
    new_content = request.form['content']
    with open(file, "w", newline="\n") as f:
        f.write(new_content)            
    return redirect(url_for('config'))