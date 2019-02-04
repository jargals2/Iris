import os

from flask import Flask

def create_app(test_config = None):
    app = Flask(__name__, instance_relative_config=True,)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    @app.route('/index')
    def index():
        return 'Hello there'
    
    from . import predict
    app.register_blueprint(predict.bp)

    return app
