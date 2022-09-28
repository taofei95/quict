from flask import Flask

from .job import job_blueprint


app = Flask(__name__)
app.register_blueprint(blueprint=job_blueprint)
