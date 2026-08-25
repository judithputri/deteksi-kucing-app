import tensorflow as tf
import numpy as np
from PIL import Image

# Muat model .h5 (dijalankan sekali saat aplikasi dibuka)
MODEL_PATH = "model_kucing.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Daftar label kelas (sesuaikan urutan dengan folder datasetmu)
CLASS_NAMES = ['Belang Tiga', 'Hitam', 'Kampung'] # Contoh nama kelas

def predict_image(image_data):
    # Preprocessing gambar sesuai format input MobileNetV2 (224x224)
    img = Image.open(image_data).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediksi
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]) * 100)
    
    return predicted_class, confidence