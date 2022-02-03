from typing import Text
from flask import Flask, render_template, request, send_file, session
from flask.helpers import url_for
from flask_sqlalchemy import SQLAlchemy
import json
import cyrtranslit
import random
import seaborn as sns
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
from werkzeug.utils import redirect
from datetime import timedelta
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report
from sklearn.tree import DecisionTreeClassifier


app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='mysql+mysqlconnector://venajankielioppi:tietokantaa@venajankielioppi.mysql.pythonanywhere-services.com/venajankielioppi$vastaus'
db=SQLAlchemy(app)
app.secret_key = 'somesecretkeythatonlyishouldknow'
app.permanent_session_lifetime = timedelta(days=1)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

class Data(db.Model):
    __tablename__="data"
    id=db.Column(db.Integer, primary_key=True)
    oikein=db.Column(db.Integer)
    muoto=db.Column(db.String)
    stud=db.Column(db.String)
    virhe=db.Column(db.String)
    def __init__(self, oikein, muoto,stud,virhe):
        self.oikein = oikein
        self.muoto = muoto
        self.stud = stud
        self.virhe = virhe
def tilasto():
    if "user" in session:
        user = session["user"]

    engine = create_engine('mysql+mysqlconnector://venajankielioppi:tietokantaa@venajankielioppi.mysql.pythonanywhere-services.com/venajankielioppi$vastaus',poolclass=NullPool, pool_recycle=280)
    table_df = pd.read_sql_table('data',engine)
    databeissihaku = sessionmaker(bind=engine)()
    table_df = table_df.replace([1],'Oikein')
    table_df = table_df.replace([0],'Virhe')

    df2 = table_df.set_index("stud",drop=False)
    df3 = df2.loc[[user],['oikein','muoto']]

    sns.catplot(y='muoto',hue="oikein",kind="count", data=df3)
    plt.savefig('/home/venajankielioppi/mysite/oikein.pdf')
    databeissihaku.close()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=["POST","GET"])
def login():
    if request.method == "POST":
        session.permanent = True
        user = request.form["nm"]
        session["user"] = user
        return redirect(url_for("user"))
    else:
        if "user" in session:
            return redirect(url_for("user"))

        return render_template("index.html")

@app.route("/logout")
def logout():
    session.pop("user",None)
    return redirect(url_for("login"))

@app.route("/tilasto")
def download():
    tilasto()
    return send_file('oikein.pdf')

@app.route("/ennuste",methods=['POST','GET'])
def ennusta():
    if "user" in session:
        user = session["user"]
    engine = create_engine('mysql+mysqlconnector://venajankielioppi:tietokantaa@venajankielioppi.mysql.pythonanywhere-services.com/venajankielioppi$vastaus',poolclass=NullPool, pool_recycle=280)
    table_df = pd.read_sql_table('data',engine)
    databeissihaku = sessionmaker(bind=engine)()

    df2 = table_df.set_index("stud",drop=False)
    df3 = df2.loc[[user],['oikein','muoto']]



    X = df3['oikein'].values.reshape(-1,1)
    y = df3['muoto'].values.reshape(-1,1)

    X.shape

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)

    classifier = LogisticRegression(random_state = 0)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    targets = ['negative class', 'positive class']
    raportti = classification_report(y_test,y_pred)
    databeissihaku.close()

    if request.method == 'GET':
        return render_template('ennuste.html')
    if request.method == 'POST':
        kielioppi = request.form['kielioppi']

        input_variables = df3['oikein'].values.reshape(-1,1)
        prediction = classifier.predict(input_variables)[1]

        return render_template('ennuste.html', original_input = {'kielioppi':'muoto'}, result = prediction)


@app.route("/user/gen")
def user():
    with open('/home/venajankielioppi/mysite/genkys.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    if "user" in session:
        user = session["user"]
        return render_template("sub.html", text = random.choice(kysymykset)), f"<h1>{user}</h1>"
    else:
        return redirect(url_for("login"))

@app.route('/user/gen', methods=['POST'])
def genvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/genkys.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        with open('/home/venajankielioppi/mysite/genvinkki.json','r',encoding='utf8') as file:
            vinkki = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Genetiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('sub.html', text = 'Oikein!')
        elif sub in vinkki:
            data=Data(0,"Genetiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('sub.html', text = vinkki[sub]['vinkki'])
        else:
            data=Data(0,"Genetiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('sub.html', text = 'Virhe!')



@app.route("/user/dat")
def dat():
    with open('/home/venajankielioppi/mysite/datkys.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    if "user" in session:
        user = session["user"]
        return render_template("erisivu.html", text = random.choice(kysymykset)), f"<h1>{user}</h1>"
    else:
        return redirect(url_for("login"))


@app.route('/user/dat', methods=['POST'])
def datvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/datkys.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        with open('/home/venajankielioppi/mysite/datvinkki.json','r',encoding='utf8') as file:
            vinkki = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Datiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('erisivu.html', text = 'Oikein!')
        elif sub in vinkki:
            data=Data(0,"Datiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('erisivu.html', text = vinkki[sub]['vinkki'])
        else:
            data=Data(0,"Datiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('erisivu.html', text = 'Virhe!')

@app.route("/user/ins")
def ins():
    with open('/home/venajankielioppi/mysite/inskys.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("sub.html", text = random.choice(kysymykset))

@app.route('/user/ins', methods=['POST'])
def insvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/inskys.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        with open('/home/venajankielioppi/mysite/insvinkki.json','r',encoding='utf8') as file:
            vinkki = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Instrumentaali",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('ins.html', text = 'Oikein!')
        elif sub in vinkki:
            data=Data(0,"Instrumentaali",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('ins.html', text = vinkki[sub]['vinkki'])
        else:
            data=Data(0,"Instrumentaali",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('ins.html', text = 'Virhe!')

@app.route("/user/pre")
def pre():
    with open('/home/venajankielioppi/mysite/prekys.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("pre.html", text = random.choice(kysymykset))

@app.route('/user/pre', methods=['POST'])
def prevas():
    if "user" in session:
        user = session["user"]

    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/prekys.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        with open('/home/venajankielioppi/mysite/vinkkipre.json','r',encoding='utf8') as file:
            vinkki = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Prepositionaali",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('pre.html', text = 'Oikein!')
        elif sub in vinkki:
            data=Data(0,"Prepositionaali",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('pre.html', text = vinkki[sub]['vinkki'])
        else:
            data=Data(0,"Prepositionaali",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('pre.html', text = 'Virhe!')

@app.route("/user/akk")
def akk():
    with open('/home/venajankielioppi/mysite/akkkys.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("akk.html", text = random.choice(kysymykset))

@app.route('/user/akk', methods=['POST'])
def akkvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/akkkys.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        with open('/home/venajankielioppi/mysite/akkvinkki.json','r',encoding='utf8') as file:
            vinkki = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Akkusatiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('akk.html', text = 'Oikein!')
        elif sub in vinkki:
            data=Data(0,"Akkusatiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('akk.html', text = vinkki[sub]['vinkki'])
        else:
            data=Data(0,"Akkusatiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('akk.html', text = 'Virhe!')

@app.route("/user/adjakk")
def ADJakk():
    with open('/home/venajankielioppi/mysite/adj_akkkys.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("adj_akk.html", text = random.choice(kysymykset))

@app.route('/user/adjakk', methods=['POST'])
def ADJakkvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/adj_akkkys.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Adj_Akkusatiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('adj_akk.html', text = 'Oikein!')
        else:
            data=Data(0,"Adj_Akkusatiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('adj_akk.html', text = 'Virhe!')

@app.route("/user/adjgen")
def ADJgen():
    with open('/home/venajankielioppi/mysite/adj_genkys.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("adj_gen.html", text = random.choice(kysymykset))

@app.route('/user/adjgen', methods=['POST'])
def ADJgenvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/adj_genkys.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Adj_Genetiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('adj_gen.html', text = 'Oikein!')
        else:
            data=Data(0,"Adj_Genetiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('adj_gen.html', text = 'Virhe!')

@app.route("/user/adjdat")
def ADJdat():
    with open('/home/venajankielioppi/mysite/adj_datkys.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("adj_akk.html", text = random.choice(kysymykset))

@app.route('/user/adjdat', methods=['POST'])
def ADJdatvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/adj_datkys.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Adj_Datiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('adj_dat.html', text = 'Oikein!')
        else:
            data=Data(0,"Adj_Datiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('adj_dat.html', text = 'Virhe!')

@app.route("/user/adjins")
def ADJins():
    with open('/home/venajankielioppi/mysite/adj_inskys.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("adj_ins.html", text = random.choice(kysymykset))

@app.route('/user/adjins', methods=['POST'])
def ADJinsvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/adj_inskys.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Adj_Instrumentaali",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('adj_ins.html', text = 'Oikein!')
        else:
            data=Data(0,"Adj_Instrumentaali",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('adj_ins.html', text = 'Virhe!')

@app.route("/user/adjpre")
def ADJpre():
    with open('/home/venajankielioppi/mysite/adj_prekys.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("adj_pre.html", text = random.choice(kysymykset))

@app.route('/user/adjpre', methods=['POST'])
def ADJprevas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/adj_prekys.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Adj_Prepositionaali",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('adj_pre.html', text = 'Oikein!')
        else:
            data=Data(0,"Adj_Prepositionaali",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('adj_pre.html', text = 'Virhe!')


@app.route("/user/prees")
def prees():
    with open('/home/venajankielioppi/mysite/preesens.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("preesens.html", text = random.choice(kysymykset))

@app.route('/user/prees', methods=['POST'])
def preesvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/preesens.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        with open('/home/venajankielioppi/mysite/preesens_vinkki.json','r',encoding='utf8') as file:
            vinkki = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Preesens",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('preesens.html', text = 'Oikein!')
        elif sub in vinkki:
            data=Data(0,"Preesens",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('preesens.html', text = vinkki[sub]['vinkki'])
        else:
            data=Data(0,"Preesens",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('preesens.html', text = 'Virhe!')

@app.route("/user/asp")
def asp():
    with open('/home/venajankielioppi/mysite/aspekti.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("aspekti.html", text = random.choice(kysymykset))

@app.route('/user/asp', methods=['POST'])
def aspvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/aspekti.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        with open('/home/venajankielioppi/mysite/aspekti_vinkki.json','r',encoding='utf8') as file:
            vinkki = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Aspekti",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('aspekti.html', text = 'Oikein!')
        elif sub in vinkki:
            data=Data(0,"Aspekti",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('aspekti.html', text = vinkki[sub]['vinkki'])
        else:
            data=Data(0,"Aspekti",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('aspekti.html', text = 'Virhe!')

@app.route("/user/pret")
def pret():
    with open('/home/venajankielioppi/mysite/preteriti.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("preteriti.html", text = random.choice(kysymykset))

@app.route('/user/pret', methods=['POST'])
def pretvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/preteriti.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        with open('/home/venajankielioppi/mysite/preteriti_vinkki.json','r',encoding='utf8') as file:
            vinkki = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Preteriti",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('preteriti.html', text = 'Oikein!')
        elif sub in vinkki:
            data=Data(0,"Preteriti",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('preteriti.html', text = vinkki[sub]['vinkki'])
        else:
            data=Data(0,"Preteriti",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('preteriti.html', text = 'Virhe!')

@app.route("/user/komp")
def komp():
    with open('/home/venajankielioppi/mysite/komparatiivi.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("komparatiivi.html", text = random.choice(kysymykset))

@app.route('/user/komp', methods=['POST'])
def kompvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/komparatiivi.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        with open('/home/venajankielioppi/mysite/komparatiivivinkki.json','r',encoding='utf8') as file:
            vinkki = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Komparatiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('komparatiivi.html', text = 'Oikein!')
        elif sub in vinkki:
            data=Data(0,"Komparatiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('komparatiivi.html', text = vinkki[sub]['vinkki'])
        else:
            data=Data(0,"Komparatiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('komparatiivi.html', text = 'Virhe!')

@app.route("/user/super")
def superla():
    with open('/home/venajankielioppi/mysite/superlatiivi.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("superlatiivi.html", text = random.choice(kysymykset))

@app.route('/user/super', methods=['POST'])
def supervas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/superlatiivi.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Superlatiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('superlatiivi.html', text = 'Oikein!')
        else:
            data=Data(0,"Superlatiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('superlatiivi.html', text = 'Virhe!')

@app.route("/user/lyhyet")
def lyh():
    with open('/home/venajankielioppi/mysite/lyhyet.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("lyhyet.html", text = random.choice(kysymykset))

@app.route('/user/lyhyet', methods=['POST'])
def lyhvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/lyhyet.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Lyhyet",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('lyhyet.html', text = 'Oikein!')
        else:
            data=Data(0,"Lyhyet",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('lyhyet.html', text = 'Virhe!')

@app.route("/user/fut")
def fut():
    with open('/home/venajankielioppi/mysite/futuuri.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("futuuri.html", text = random.choice(kysymykset))

@app.route('/user/fut', methods=['POST'])
def futvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/futuuri.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Futuuri",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('futuuri.html', text = 'Oikein!')
        else:
            data=Data(0,"Futuuri",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('futuuri.html', text = 'Virhe!')

@app.route("/user/num")
def num():
    with open('/home/venajankielioppi/mysite/numerot.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("numerot.html", text = random.choice(kysymykset))

@app.route('/user/num', methods=['POST'])
def numvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/numerot.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        with open('/home/venajankielioppi/mysite/numerot_vinkki.json','r',encoding='utf8') as file:
            vinkki = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Numerot",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('numerot.html', text = 'Oikein!')
        elif sub in vinkki:
            data=Data(0,"Numerot",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('superlatiivi.html', text = vinkki[sub]['vinkki'])
        else:
            data=Data(0,"Numerot",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('numerot.html', text = 'Virhe!')

@app.route("/user/pas")
def pas():
    with open('/home/venajankielioppi/mysite/passiivi.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("passiivi.html", text = random.choice(kysymykset))

@app.route('/user/pas', methods=['POST'])
def pasvas():
    if "user" in session:
        user = session["user"]
    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/passiivi.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Passiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('passiivi.html', text = 'Oikein!')
        else:
            data=Data(0,"Passiivi",user,sub)
            db.session.add(data)
            db.session.commit()
            return render_template('passiivi.html', text = 'Virhe!')



if __name__=="__main__":
    app.run(debug=False)
