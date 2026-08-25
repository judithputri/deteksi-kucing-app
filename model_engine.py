import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load model TFLite (ringan & kompatibel dengan Python 3.14)
MODEL_PATH = "model_kucing.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ['Belang Tiga', 'Hitam', 'Kampung'] # Sesuaikan nama kelasmu

def predict_image(image_data):
    # Preprocessing gambar
    img = Image.open(image_data).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi via TFLite
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]) * 100)

    return predicted_class, confidence
