from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask.ext.wtf import Form
from wtforms import TextField, SubmitField
from wtforms import validators
import pickle


#Initialize Flask App
app = Flask(__name__)

#Initialize Form Class
# This form will take in the form data on the front end and use it to predict
# using a pre-loaded model
class PredictForm(Form):
    sentence = TextField('', validators=[validators.required(),
                                         validators.length(max=200)])
    submit   = SubmitField('Submit')

# unpickle my model and load it in memory
model, target_names = pickle.load(open('ml/model.pkl'))
print "Model loaded in memory. Ready to roll!"

def lang_attr(x):
    return {
        'Ar': ['Arabic', 'Arabic-flag.png'],
        'Bg': ['Bulgarian', 'Bulgaria-flag.png'],
        'De': ['German', 'Germany-flag.png'],
        'En': ['English', 'USA-flag.png'],
        'Es': ['Spanish', 'Spain-flag.png'],
        'Fr': ['French', 'France-flag.png'],
        'It': ['Italian', 'Italy-flag.png'],
        'Ja': ['Japanese', 'Japan-flag.png'],
        'Nl': ['Dutch', 'Dutch-flag.png'],
        'Pl': ['Polish', 'Poland-flag.png'],
        'Pt': ['Portuguese', 'Portugal-flag.png'],
        'Ru': ['Russian', 'Russia-flag.png'],
    }.get(x, ['No language match', 'no-flag.png'])

@app.route('/',methods=['GET', 'POST'])
def translate():
    
    prediction, prediction_flag, sentence = None, None, None
    predict_form = PredictForm(csrf_enabled=False)

    if predict_form.validate_on_submit():

        # store the submitted values
        submitted_data = predict_form.data
        print submitted_data

        # Retrieve values from form
        sentence = submitted_data['sentence']

        # Predict the class corresponding to the sentence
        predicted_class_n = model.predict([sentence])

        # Get the corresponding class name and make it pretty
        prediction = lang_attr(target_names[predicted_class_n].capitalize())[0]
        prediction_flag = lang_attr(target_names[predicted_class_n].capitalize())[1]


    # Pass the predicted class name to the fron-end
    return render_template('model.html',
                            predict_form = predict_form, 
                            prediction   = prediction,
			    prediction_flag = prediction_flag)

#Handle Bad Requests
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
