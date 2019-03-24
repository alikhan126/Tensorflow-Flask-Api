from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
from threading import Thread


from flask import Flask,request
from flask import json
import requests
import time

app = Flask(__name__)


models = ['flowers','cats_dogs','computer_accessories']
graphs = dict()

def load_graphs():
	for model in models:
	    graph = tf.Graph()
	    graph_def = tf.GraphDef()

	    with open('./models/'+model+'.pb', "rb") as f:
	        graph_def.ParseFromString(f.read())
	    with graph.as_default():
	        tf.import_graph_def(graph_def)

	    graphs[model] = graph

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def download(imgurl):
	import datetime
	imgfilename = datetime.datetime.today().strftime('%Y%m%d') + '_' + imgurl.split('/')[-1]
	with open("uploads/" + imgfilename, 'wb') as f:
		f.write(requests.get(imgurl).content)
	return imgfilename;


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):


    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result

def proccess_image(result,graph,model,file_name):
	t = read_tensor_from_image_file(file_name,input_height=224,input_width=224,input_mean=127.5,input_std=127.5)

	input_operation = graph.get_tensor_by_name(input_name)
	output_operation = graph.get_tensor_by_name(output_name)

	with tf.Session(graph=graph) as sess:
	    results = sess.run(output_operation,
	                       {input_name: t})
	results = np.squeeze(results)

	top_k = results.argsort()[-3:][::-1]
	labels = load_labels('./models/'+model+'_labels.txt')
	data = []
	for i in top_k:
	    data.append({'label':labels[i], 'confidence' : float(results[i])})
	result[model] = sorted(data, key=lambda x: x['confidence'],reverse=True)

#load graphs
load_graphs();

input_layer = "input:0"
output_layer = "final_result:0"
input_name = "import/" + input_layer
output_name = "import/" + output_layer

@app.route("/", methods = ['GET'])
def predict():

	if(request.args.get('image_url') == None):
		return app.response_class(
		    response= json.dumps({'msg' : 'Please provide "image_url" query parameter i-e ?image_url=https://i.imgur.com/oDf68ZO.jpg'}),
		    status=500,
		    mimetype='application/json'
		)

	threads= []

	image_url = request.args.get('image_url')
	file_name = "uploads/" + download(image_url);

	start = time.clock()

	result = dict()
	for model in models:
		graph = graphs[model]

		process = Thread(target=proccess_image, args=[result,graph,model,file_name])
		process.start()
		threads.append(process)

	for process in threads:
		process.join()
		
	request_time = time.clock() - start
	result['response_time'] = str(round(request_time,2)) + "sec"

	response = app.response_class(
	    response=json.dumps(result),
	    status=200,
	    mimetype='application/json'
	)
	return response

app.run()