from flask import Flask
from flask import Flask, send_from_directory, render_template, Response, request, make_response, session
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
from werkzeug.utils import secure_filename
from datetime import datetime, date, timedelta
import random
import string
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import gc
import sys
import time
import datetime
import lightgbm as ltb
import random
from sklearn.model_selection import train_test_split

app = Flask(__name__, static_folder="templates", template_folder='templates')


@app.route('/')
def start():
    return render_template('/start.html')


@app.route('/index.html')
def index():
    return render_template('/index.html')


@app.route('/si.html')
def si():
    return render_template('/si.html')


@app.route('/train.html', methods=['POST', 'GET'])
def train():
    if request.method == 'POST':
        results = request.form.to_dict()
        if results['username'] == "manojainala" and results['password'] == "manoj@123":
            # %% [code]
            if len(results) == 2:

                return render_template('/train.html', showlogin=0, message="", results=results)

            f = request.files['fileToUpload']
            filePath = f.filename
            print(filePath)
            f.save(secure_filename(filePath))
            # %% [code]
            df = pd.read_csv(filePath)

            X = df[['HISPORX', 'SEX', 'DIABETES', 'HYPERTEN', 'HYPERCHO', 'TOBAC30',
                    'ALCOHOL', 'NACCMMSE', 'CDRGLOB', 'CVHATT', 'NACCALZD', 'CANCER']]
            y = df['Impairment']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25)

            train_set = ltb.Dataset(X_train, y_train)
            valid_set = ltb.Dataset(X_test, y_test)

            params = {
                "objective": "multiclass",
                "metric": "multi_error",
                "learning_rate": 0.03,
                'num_class': 5
            }
            evals_result = {}  # to record eval results for plotting

            model = ltb.train(params,
                              train_set=train_set,
                              num_boost_round=10000,
                              early_stopping_rounds=200,
                              verbose_eval=100,
                              evals_result=evals_result,
                              valid_sets=[train_set, valid_set]
                              )
            model.save_model('model.txt')
            return render_template('/index.html', showlogin=0, message="")
        else:
            return render_template('/train.html', showlogin=1, message="Wrong Username or Password")
    else:
        return render_template('/train.html', showlogin=1, message="")


@app.route('/predict.html', methods=['POST', 'GET'])
def predict():
    print("checking predict!!")
    if request.method == 'POST':
        print("checking predict!!", file=sys.stderr)
        Results = request.form.to_dict()
        for Result in Results:
            if Results[Result] == "":
                Results[Result] = 0

            try:
                Results[Result] = float(Results[Result])
            except:
                pass

        print(Results)
        df = pd.DataFrame([Results])
       
        print(df, file=sys.stderr)
        print(df.head())

        model = ltb.Booster(model_file='model.txt')
        y_pred = model.predict(df)
        y_pred[0][1] = round(round(y_pred[0][1]*100, 2)/100, 2)
        y_pred[0][2] = round(round(y_pred[0][2]*100, 2)/100, 2)
        y_pred[0][3] = round(round(y_pred[0][3]*100, 2)/100, 2)
        y_pred[0][4] = round(round(y_pred[0][4]*100, 2)/100, 2)
        print(y_pred)
        return render_template('/Results.html', pred=y_pred[0])
    else:
        return render_template('/predict.html')


@app.route('/visualization1.html', methods=['POST', 'GET'])
def visualization1():
    if request.method == 'POST':
        print("OK! coming")
        results = request.form.to_dict()
        print("----")
        print(results)
        print("----")
        df = pd.read_csv('manoz.csv')

        mmse = int(results['NACCMMSE'])
        cdr = float(results['CDRGLOB'])

        sex = int(results['SEX'])
        

        df2 = df[(df.NACCMMSE == mmse) & (df.CDRGLOB == cdr) & (df.SEX == sex)]
       
        df2.groupby(['CDRGLOB', 'SEX', 'NACCMMSE', 'Impairment']
                    ).size().unstack().plot(kind='bar', stacked=True)
        plt.ylabel("Count of persons with different impairments in total data")
        plt.xlabel("CDR,SEX,MMSE")

        # os.remove('templates/checkfig.png')
        basename = "templates/figure"
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        # e.g. 'mylogfile_120508_171442'
        filename = "_".join([basename, suffix])
        plt.savefig(filename+'.png', bbox_inches='tight')
        print(filename)

     
        time.sleep(10)
        return render_template('/visualization1.html', query_ret=1, filename=filename+'.png')
    else:
        return render_template('/visualization1.html', query_ret=0)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
