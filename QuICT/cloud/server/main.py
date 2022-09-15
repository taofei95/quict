from flask import Flask

from .job import job_blueprint
from .cluster import cluster_blueprint


app = Flask(__name__)
app.register_blueprint(blueprint=job_blueprint)
app.register_blueprint(blueprint=cluster_blueprint)