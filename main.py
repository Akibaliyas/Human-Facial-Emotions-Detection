from mtcnn import MTCNN
import cv2
import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import base64
import time

# PDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# -------------------------------------------------
# Disable oneDNN warning
# -------------------------------------------------
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# -------------------------------------------------
# Flask App
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'facial_emotion_detection_model.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------
# Load Model & Detector
# -------------------------------------------------
model = load_model(MODEL_PATH)
detector = MTCNN()

# -------------------------------------------------
# Emotion Data
# -------------------------------------------------
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

emotion_emojis = {
    'happy': '😊', 'sad': '😢', 'angry': '😠',
    'fear': '😨', 'disgust': '🤢',
    'neutral': '😐', 'surprise': '😲'
}

emotion_themes = {
    'happy': '#FFF9C4', 'sad': '#BBDEFB', 'angry': '#FFCDD2',
    'fear': '#E1BEE7', 'disgust': '#C8E6C9',
    'neutral': '#EEEEEE', 'surprise': '#FFE0B2'
}

emotion_quotes = {
    'happy': "Happiness comes from your actions.",
    'sad': "Tough times don’t last.",
    'angry': "Anger harms the holder.",
    'fear': "Fear is temporary.",
    'disgust': "Protect your peace.",
    'neutral': "Calm brings clarity.",
    'surprise': "Unexpected can be good."
}

emotion_tips = {
    'happy': "Share positivity.",
    'sad': "Talk to someone you trust.",
    'angry': "Pause and breathe.",
    'fear': "Face fears slowly.",
    'disgust': "Practice self-care.",
    'neutral': "Reflect and recharge.",
    'surprise': "Stay open-minded."
}

# -------------------------------------------------
# PDF GENERATOR (FORM STYLE + IMAGE)
# -------------------------------------------------
def generate_pdf(emotion, confidence, image_path):
    pdf_name = f"session_report_{int(time.time())}.pdf"
    pdf_path = os.path.join(STATIC_FOLDER, pdf_name)

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 50, "Facial Emotion Detection Report")

    # Border (Form Look)
    c.rect(40, 40, width - 80, height - 120)

    # Labels
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, height - 110, "Session Details")

    c.setFont("Helvetica", 11)
    c.drawString(60, height - 150, "Detected Emotion:")
    c.drawString(60, height - 180, "Confidence Score:")
    c.drawString(60, height - 210, "Quote:")
    c.drawString(60, height - 250, "Mental Health Tip:")

    # Values
    c.setFont("Helvetica-Bold", 11)
    c.drawString(200, height - 150, f"{emotion} {emotion_emojis[emotion]}")
    c.drawString(200, height - 180, f"{confidence}%")

    c.setFont("Helvetica", 11)
    c.drawString(200, height - 210, emotion_quotes[emotion])
    c.drawString(200, height - 250, emotion_tips[emotion])

    # Image Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, height - 310, "Analyzed Image:")

    try:
        img = ImageReader(image_path)
        c.drawImage(img, 60, height - 550, width=200, height=200, preserveAspectRatio=True)
    except:
        c.drawString(60, height - 340, "Image not available")

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(
        width / 2, 60,
        "Developed by Aqib Ali Ilyas & Zakir Hussain | AI Emotion Detection System"
    )

    c.showPage()
    c.save()

    return pdf_name

# -------------------------------------------------
# Emotion Detection
# -------------------------------------------------
def detect_emotion(img_path):
    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb)
    if not faces:
        return None, None, "No face detected"

    face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
    x, y, w, h = face['box']
    x, y = max(0, x), max(0, y)

    face_img = rgb[y:y+h, x:x+w]
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (48, 48)) / 255.0
    gray = gray.reshape(1, 48, 48, 1)

    pred = model.predict(gray, verbose=0)
    idx = np.argmax(pred)

    emotion = class_names[idx]
    confidence = round(float(pred[0][idx]) * 100, 2)

    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"{emotion_emojis[emotion]} {emotion} ({confidence}%)",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imwrite(img_path, img)
    return emotion, confidence, None

# -------------------------------------------------
# Route
# -------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

        elif 'webcam_image' in request.form:
            data = request.form['webcam_image']
            img_data = base64.b64decode(data.split(',')[1])
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

            filename = f"webcam_{int(time.time())}.jpg"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            cv2.imwrite(file_path, img)

        else:
            return render_template("index.html", error="No image provided")

        emotion, confidence, error = detect_emotion(file_path)
        if error:
            return render_template("index.html", error=error)

        pdf_file = generate_pdf(emotion, confidence, file_path)

        return render_template(
            "index.html",
            image_path=f"uploads/{filename}",
            emotion=emotion,
            confidence=confidence,
            quote=emotion_quotes[emotion],
            tip=emotion_tips[emotion],
            theme_color=emotion_themes[emotion],
            pdf_file=pdf_file
        )

    return render_template("index.html")

# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
