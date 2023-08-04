import os
import yaml

from flask import Flask, request, render_template, send_file
from zipfile import ZipFile
from io import BytesIO

from bowel.models.predict import Inference
from bowel.reporting.statistics import Report
from bowel.utils.train_utils import convert_to_sounds

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024 # 10GB

recording = 'recording.wav'
sounds_output = 'sounds.csv'
report_output = 'report.xlsx'
output = 'results.zip'

model_dir = 'webserver/models'
model = 'crnn03052023'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods = ['POST'])
def upload():
    if request.method == 'POST':
        request.files['file'].save(recording)
        return {}


@app.route('/inference', methods = ['POST'])
def inference():
    try:
        model_path = os.path.join(model_dir, model + '.h5')
        config = yaml.safe_load(open(os.path.join(model_dir, model + '.yml')))
        inference_model = Inference(model_path, config)
        frames, duration = inference_model.infer(recording)
        sounds = convert_to_sounds(frames)
        inference_model.save_predictions(sounds_output, sounds, duration)
        report = Report(sounds_output)
        report.save(report_output)
        error = ''
        result = 200
    except Exception as e:
        error = str(e) or 'Failed to infer the file'
        result = 500
    return {'error': error, 'result': result}


@app.route('/download', methods = ['GET'])
def download():
    try:
        stream = BytesIO()
        with ZipFile(stream, 'w') as zf:
            zf.write(sounds_output)
            zf.write(report_output)
        stream.seek(0)
    except Exception:
        return {}
    return send_file(stream, as_attachment=True, download_name=output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
