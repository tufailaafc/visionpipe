from flask import Flask, Response
import cv2

app = Flask('mjpeg')

#Return frames generated from the rtsp stream, specifically from deepstream but could be extended later.
def gen_frames():
    cap = cv2.VideoCapture("rtsp://<deepstream-host>:8554/ds-test")
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


# route to get the series of images
@app.route('/stream.mjpeg')
def stream():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8801)
