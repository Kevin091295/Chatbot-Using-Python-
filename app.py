from flask import Flask, render_template, request
from chatbot_logic import get_response, faq_dict, embeddings  # Importing chatbot logic

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form["user_input"]
        response = get_response(user_input, faq_dict, embeddings)
        return render_template("chat.html", user_input=user_input, response=response)
    return render_template("chat.html")

if __name__ == "__main__":
    app.run(debug=True)
