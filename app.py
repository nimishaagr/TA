#FLASK
from flask import Flask, render_template,request

app = Flask(__name__)
from textblob import TextBlob
from transformers import pipeline

classifier = pipeline('sentiment-analysis', "mrm8488/bert-small-finetuned-squadv2")

@app.route("/", methods=["GET", "POST"]) #even though only using post, best practice to have both
def index():
  if request.method == "POST": #after you press submit button
    text = request.form.get("text")
    print(text)
    r1 = TextBlob(text).sentiment
    r2 = classifier(text)
    return(render_template("index.html", result1=r1, result2=r2))
  else: #before pressing submit button
    return(render_template("index.html", result1="2", result2="2"))

if __name__=="__main__": #not needed on locall computer, but necessary for cloud to confirm you're the author
  app.run() #in development environment