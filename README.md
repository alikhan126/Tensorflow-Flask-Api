# Tensorflow image image recognition rest api
**Install requirments**

`pip -r requirements.txt`

**Train your dataset**

Put your dataset into 'dataset' folder, and dataset must have sub-folders for categories.

    python retrain.py --image_dir=dataset/<dataset-name> --how_many_training_steps=500  --architecture=mobilenet_1.0_224 --output_graph=models/<model-name>.pb --output_labels=models/<model-name>_labels.txt

Edit  *`server.py`* file , and add your model-name into graph list at *line 20*.

Run command  `python server.py`

**Test**

http://127.0.0.1:5000/?image_url=https://i.imgur.com/oDf68ZO.jpg