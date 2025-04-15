from flask import render_template

def render_chat_page(theme="light"):
    return render_template("index.html", theme=theme)