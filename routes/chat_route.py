from flask import render_template
from opennlu import app
    
@app.route('/chat')
def chat():
    return render_template('/chat/index.html')