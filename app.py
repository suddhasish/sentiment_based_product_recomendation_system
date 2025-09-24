# app.py (minimal)
from flask import Flask, render_template, request
from model import recommend_for_user

app = Flask(__name__)
app.secret_key = "replace-me"

@app.route("/", methods=["GET", "POST"])
def index():
    username = ""
    recs = []
    if request.method == "POST":
        username = request.form.get("username","").strip()
        if username:
            recs = recommend_for_user(username, top_n=5)
    return render_template("index.html", username=username, recommendations=recs)

if __name__ == "__main__":
    app.run(debug=True)
