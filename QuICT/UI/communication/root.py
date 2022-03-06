from flask import Flask

from QuICT.UI.communication.blueprint.composer_run import blueprint as composer_blueprint
# from QuICT.UI.communication.blueprint.qcda_run import blueprint as qcda_blueprint

# App related
app = Flask(__name__)
app.url_map.strict_slashes = False

app.register_blueprint(blueprint=composer_blueprint)
# app.register_blueprint(blueprint=qcda_blueprint)

app.run(host="0.0.0.0", debug=True)
