import os
import json
# from spark import recommender
from flask import Flask, request
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
		
@app.route('/predict',methods=['POST'])
def predict():
    data = request.json
    reco.add_new_user(data['new_user_ratings'],data['new_user_id'])
    logging.info('New user added')
    reco.train_model()
    logging.info('Model trained')
    reco.make_recomendations()
    logging.info('Making and showing recomendations')
    reco.show_recomendations()
    return 'Ok'

if __name__ == '__main__':
	reco = recommender()
	logging.info('Recommender initialised')
	reco.start_spark()
	logging.info('Spark booted')
	reco.read_data()
	logging.info('Data read')
	app.run(host='0.0.0.0')
	