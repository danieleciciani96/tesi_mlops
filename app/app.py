import os
from datetime import datetime
from flask import Flask, render_template, jsonify


STATIC_PATH = "reports/"
TEMPLATE_PATH = "apps/templates"

app = Flask(__name__, static_folder=STATIC_PATH, template_folder=TEMPLATE_PATH)


@app.route("/")
def hello_world():
    return render_template("/home/cdsw/reports/report_data_target_drift.html")


"""
@app.route("/get_report_dates", methods=["GET"])
def get_report_dates():
    report_dates = sorted(
        os.listdir(os.path.join(STATIC_PATH, "reports")),
        key=lambda date: datetime.strptime(date.split("_")[0], "%m-%d-%Y"),
        reverse=True,
    )
    return jsonify(report_dates)
"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=os.environ.get("CDSW_READONLY_PORT"))