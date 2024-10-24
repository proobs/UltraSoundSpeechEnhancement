import os
from flask import Flask, request, jsonify, send_from_directory
from pydub import AudioSegment
from scipy.signal import butter, lfilter
import numpy as np

app = Flask(__name__)

# Define the upload folder inside the same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'upload')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file types
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def split_audio(file_path):
    sound = AudioSegment.from_file(file_path)

    # Convert AudioSegment to raw data
    samples = np.array(sound.get_array_of_samples())
    sample_rate = sound.frame_rate

    normal_filtered = butter_bandpass_filter(samples, 20, 20000, sample_rate)
    ultrasound_filtered = butter_bandpass_filter(
        samples, 20000, 96000, sample_rate)

    normal_sound = AudioSegment(
        normal_filtered.astype(np.int16).tobytes(),
        frame_rate=sample_rate,
        sample_width=sound.sample_width,
        channels=sound.channels
    )

    ultrasound_sound = AudioSegment(
        ultrasound_filtered.astype(np.int16).tobytes(),
        frame_rate=sample_rate,
        sample_width=sound.sample_width,
        channels=sound.channels
    )

    # Export the split files
    ultrasound_file_path = file_path.replace(".wav", "_ultrasound.wav")
    normal_file_path = file_path.replace(".wav", "_normal.wav")

    ultrasound_sound.export(ultrasound_file_path, format="wav")
    normal_sound.export(normal_file_path, format="wav")

    return ultrasound_file_path, normal_file_path


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        ultrasound_file_path, normal_file_path = split_audio(file_path)

        return jsonify({
            'message': 'File successfully uploaded and split',
            'ultrasound_file': ultrasound_file_path,
            'normal_file': normal_file_path
        }), 200

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/upload/<filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
