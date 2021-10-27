from typing import Text
from flask import Flask, render_template, request, send_file, session, g
from flask.helpers import url_for
from flask_sqlalchemy import SQLAlchemy
import json
import cyrtranslit
import random
import seaborn as sns
import pandas as pd
import sqlalchemy
from matplotlib.figure import Figure
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
from werkzeug.utils import redirect
from datetime import timedelta
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker

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
    def __init__(self, oikein, muoto,stud):
        self.oikein = oikein
        self.muoto = muoto
        self.stud = stud

def tilasto():
    if "user" in session:
        user = session["user"]

    engine = create_engine('mysql+mysqlconnector://venajankielioppi:tietokantaa@venajankielioppi.mysql.pythonanywhere-services.com/venajankielioppi$vastaus',poolclass=NullPool, pool_recycle=280)
    table_df = pd.read_sql_table('data',engine)
    databeissihaku = sessionmaker(bind=engine)()
    table_df = table_df.replace([1],'Oikein')
    table_df = table_df.replace([0],'Virhe')
    table_df = table_df.replace([2],'Vinkki')

    df2 = table_df.set_index("stud",drop=False)
    df3 = df2.loc[[user],['oikein','muoto']]

    sns.catplot(y='oikein',kind="count", data=df3)
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
def vastaa():
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
            data=Data(1,"Genetiivi",user)
            db.session.add(data)
            db.session.commit()
            return render_template('sub.html', text = 'Oikein!')
        elif sub in vinkki:
            return render_template('sub.html', text = vinkki[sub]['vinkki'])
        else:
            data=Data(0,"Genetiivi",user)
            db.session.add(data)
            db.session.commit()
            return render_template('sub.html', text = 'Virhe!')

@app.route("/user/tilasto")
def download():
    tilasto()
    return send_file('oikein.pdf')


@app.route("/user/datiivi")
def erisivu():
    with open('/home/venajankielioppi/mysite/datkys.txt','r',encoding='utf8') as file:
        kysymykset = file.readlines()
    return render_template("erisivu.html", text = random.choice(kysymykset))



@app.route('/user/datiivi', methods=['POST'])
def dat():

    if request.method=="POST":
        sub=request.form['vastaus']
        with open('/home/venajankielioppi/mysite/datkys.json','r',encoding='utf8') as file:
            substantiivi = json.load(file)
        with open('/home/venajankielioppi/mysite/datvinkki.json','r',encoding='utf8') as file:
            vinkki = json.load(file)
        sub = cyrtranslit.to_cyrillic(sub.lower(),'ru')
        if sub in substantiivi:
            data=Data(1,"Datiivi",user)
            db.session.add(data)
            db.session.commit()
            return render_template('erisivu.html', text = 'Oikein!')
        elif sub in vinkki:
            return render_template('erisivu.html', text = vinkki[sub]['vinkki'])
        else:
            data=Data(0,"Datiivi",user)
            db.session.add(data)
            db.session.commit()
            return render_template('erisivu.html', text = 'Virhe!')




if __name__=="__main__":
    app.run(debug=False)
