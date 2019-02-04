import functools


from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
import pickle

bp = Blueprint('predict', __name__, url_prefix = '/api')

@bp.route('/predict', methods=('GET','POST'))
def predict():
    
    prediction = ""
    if request.method == 'GET':
        sep_len = request.args.get('sl')
        sep_wid = request.args.get("sw")
        pet_len = request.args.get('pl')
        pet_wid = request.args.get('pw')
        

        error = None
        if not (sep_len and sep_wid and pet_len and pet_wid):
            error = "One of the fields is missing"
            flash(error)
        else:
            #load the model and predict with the given parameters
            feature_array = [[sep_len, sep_wid, pet_len, pet_wid]]
            knn_model = pickle.load(open('Iris/knn_model.pkl', 'rb'))
            prediction = knn_model.predict(feature_array)
    
    return render_template("/predict.html", species = prediction)
