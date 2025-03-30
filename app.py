from flask import Flask, request, jsonify, render_template
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('final_model_ishu3.h5')

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded

@app.route('/predict', methods=['POST'])
def predict():
    if 'frame' in request.files:
        filestr = request.files['frame'].read()
        nparr = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed_frame = preprocess_frame(image)
        prediction = model.predict(processed_frame)
        emotion = np.argmax(prediction)
        label_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'}
        emotion_prediction = label_map.get(emotion, 'Unknown')
        return jsonify({'emotion': emotion_prediction})

    return jsonify({'error': 'No frame received'}), 400

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
