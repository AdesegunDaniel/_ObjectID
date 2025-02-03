from flask import Flask, request, send_file, redirect, url_for, render_template, jsonify
from flask_socketio import SocketIO, emit
from io import BytesIO
import os
import cv2
import base64
import numpy as np
import shutil
import time
from ultralytics import YOLO


# Load the model
model = YOLO("yolo11n.pt")
# Perform inference on a video

# The processed video will be saved automatically in the 'runs/detect' directory

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@app.route('/') 
def obj_home(): 
    return render_template('obj_home.html')

# Main function to determine input type and process accordingly
@app.route('/process_picture', methods=['POST']) 
def process_picture(model=model):
    input_type = request.form.get('inputType')
    
    if input_type == 'live':
        # Render the index template for live camera option
        return render_template('index.html')
    
    file = request.files['file']
    folder = 'static'
    input_path = os.path.join(folder, file.filename)
    
    print('process has started')
    
    if input_path.endswith(('.jpg', '.jpeg', '.png')):
        # Process the image
        results = model(input_path)
        # Define the output path 
        output_path = 'static/processed_image.jpg'
        results[0].save(output_path)

        time.sleep(1)
        return render_template('result1.html', resultType='image')
    
    elif input_path.endswith(('.mp4', '.avi', '.mov')):
        # Process the video
        # Load the input video to get its frame rate
        input_video = cv2.VideoCapture(input_path)
        input_frame_rate = input_video.get(cv2.CAP_PROP_FPS)
        results = model(input_path)
        # Define the output path 
        output_path = 'static/processed_image.mp4'
        annotated_frames = []
        for result in results:
            annotated_frame = result.plot()  # Get the annotated frame
            annotated_frames.append(annotated_frame)

        # Define the output video path and settings
        height, width, layers = annotated_frames[0].shape
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'H264'), input_frame_rate, (width, height))
        # Write each annotated frame to the video file
        if not video_writer.isOpened():
            print("Error: Could not open video writer")
        else:
            print("Video writer opened successfully")

        for frame in annotated_frames:
            video_writer.write(frame)

        # Release the video writer
        video_writer.release()

        time.sleep(1)
        
        print('all done')
        return render_template('result2.html', resultType='video')


@socketio.on('process_image')
def process_image(data):
    image_data = data['image'].split(',')[1]
    image = base64.b64decode(image_data)
    np_image = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    results = model(img)
    plot_image = results[0].plot()

    _, buffer = cv2.imencode('.jpg', plot_image)
    processed_image = base64.b64encode(buffer).decode('utf-8')

    emit('processed_image', {'processed_image': processed_image})


@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['file']
    folder = 'static'
    if os.path.exists(folder):
        shutil.rmtree(folder)  # Remove the existing folder and its contents
    os.makedirs(folder)  # Create a new, empty folder
    if file:
        # Save the uploaded file to a temporary location
        file_path = os.path.join(folder, file.filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'file_path': file_path})
    else: 
        return jsonify({'error': 'No file uploaded'}), 400
    

@app.route('/stop', methods=['POST'])
def stop():
    folder = 'static'
    if os.path.exists(folder):
        shutil.rmtree(folder) 
    return render_template('obj_home.html')


if __name__ == '__main__': 
    socketio.run(app, host='0.0.0.0', port=81)
