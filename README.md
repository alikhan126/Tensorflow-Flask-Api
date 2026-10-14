# Tensorflow-Flask-Api

A small Flask API (2018) that classifies an image from a URL with several
TensorFlow image classifiers at once. Each model is a MobileNet that was retrained
on a custom dataset with TensorFlow's transfer-learning script (`retrain.py`).

## How it works

1. You call `GET /?image_url=<url>`.
2. The server downloads the image into `uploads/`.
3. Every model in `models/` classifies it, each in its own thread.
4. The API returns the top 3 labels and their confidence for each model, plus the
   total response time.

## Included models

| Model | Labels |
|---|---|
| `flowers` | daisy, dandelion, roses, sunflowers, tulips |
| `cats_dogs` | cat, dog |
| `computer_accessories` | gamingcontroller, keyboard, mouse, mousepen |

## Running it

Written for **Python 3.6 and TensorFlow 1.8** (it uses the TF1 APIs `tf.GraphDef`,
`tf.Session` and `tf.gfile`, and `time.clock`, which was removed in Python 3.8).
It does not run on TensorFlow 2 or newer Python versions unchanged.

```
pip install -r requirements.txt
mkdir uploads
python server.py
```

Test it:

```
curl "http://127.0.0.1:5000/?image_url=https://i.imgur.com/oDf68ZO.jpg"
```

Example response shape:

```json
{
  "flowers": [{"label": "roses", "confidence": 0.91}, ...],
  "cats_dogs": [{"label": "cat", "confidence": 0.62}, ...],
  "computer_accessories": [{"label": "mouse", "confidence": 0.40}, ...],
  "response_time": "0.85sec"
}
```

## Training your own model

Put your images in `dataset/<dataset-name>/`, with one sub-folder per category, then run:

```
python retrain.py --image_dir=dataset/<dataset-name> \
  --how_many_training_steps=500 \
  --architecture=mobilenet_1.0_224 \
  --output_graph=models/<model-name>.pb \
  --output_labels=models/<model-name>_labels.txt
```

Then add `<model-name>` to the `models` list in `server.py`.

## Status

Unmaintained, kept for reference. It's a demo: the server downloads any URL it
is given and has no input validation, so don't expose it to the internet.
