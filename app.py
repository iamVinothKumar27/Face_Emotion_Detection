from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your emotion detection model
model = tf.keras.models.load_model('final_model_ishu3.h5')

def preprocess_frame(frame):
    # Resize frame to fit model's expected input
    frame_resized = cv2.resize(frame, (224, 224))  # Example size, adjust as necessary
    # Convert color from BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    # Normalize pixel values if your model expects normalization
    frame_normalized = frame_rgb / 255.0
    # Expand dimensions to match the model's input format
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded

def gen_frames():
    camera = cv2.VideoCapture(0) 
    if not camera.isOpened():
        raise RuntimeError('Could not start camera.')
    
    while True:
        success, frame = camera.read()
        if not success:
            break

        processed_frame = preprocess_frame(frame)
        prediction = model.predict(processed_frame)
        
        print("Processed Frame Shape:", processed_frame.shape)  # Debugging
        print("Raw Model Prediction:", prediction)  # Debugging

        emotion = np.argmax(prediction)
        label_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad',6:'surprised'}
        emotion_prediction = label_map.get(emotion, 'Unknown')

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, emotion_prediction, (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_4)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
