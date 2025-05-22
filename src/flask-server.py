import json

from flask import Flask, render_template, request

from src.core.core import construct_graph, explain, read_model

app = Flask(__name__)

model = read_model()


@app.route("/")
def inputForm():
    return render_template('form.html')


@app.route("/report", methods=['POST'])
def generateReport():
    title = request.form['title']
    content = request.form['content']

    data = construct_graph(title, content)
    report = explain(model, data, "test")
    report['title'] = title
    report['content'] = content

    return render_template('report_begin.html') + json.dumps(report) + render_template('report_end.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
