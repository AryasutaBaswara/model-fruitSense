import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image

# --- 1. Konfigurasi Nama File (Sesuai Screenshot Anda) ---
# File berada satu folder dengan script ini, jadi tidak perlu path folder
IDENTIFIER_MODEL_PATH = 'identifier_model_fruitsense.tflite'
GRADING_MODEL_PATH = 'grading_model_fruitsense.tflite'

IMAGE_SIZE = 224 # Ukuran input standar MobileNetV2

print(f"Sedang memuat model dari: {IDENTIFIER_MODEL_PATH} dan {GRADING_MODEL_PATH}...")

# --- 2. Muat Model TFLite ---
try:
    # Load Model 1: Identifikasi
    interpreter_identifier = tf.lite.Interpreter(model_path=IDENTIFIER_MODEL_PATH)
    interpreter_identifier.allocate_tensors()
    
    # Load Model 2: Grading
    interpreter_grading = tf.lite.Interpreter(model_path=GRADING_MODEL_PATH)
    interpreter_grading.allocate_tensors()

    # Dapatkan detail input/output untuk inferensi
    id_input_details = interpreter_identifier.get_input_details()
    id_output_details = interpreter_identifier.get_output_details()
    
    grade_input_details = interpreter_grading.get_input_details()
    grade_output_details = interpreter_grading.get_output_details()
    
    print("✅ Berhasil memuat kedua model TFLite!")

except Exception as e:
    print(f"❌ Gagal memuat model. Pastikan nama file benar.\nError: {e}")
    exit()

# --- 3. Definisikan Label (PENTING: URUTAN HARUS SAMA DENGAN COLAB) ---

# Label Model 1 (11 Buah - Urutan Alfabet Inggris biasanya)
# Pastikan urutan ini SAMA PERSIS dengan output: train_generator.class_indices di Colab
IDENTIFIER_LABELS = [
    'apple', 'banana', 'durian', 'grape', 'guava', 
    'jackfruit', 'mango', 'orange', 'papaya', 'pineapple', 'watermelon'
] 

# Label Model 2 (44 Kelas Grading)
# ANDA HARUS MENGISI INI MANUAL SESUAI URUTAN DI COLAB ANDA
# Buka Colab Grading Anda, jalankan: print(train_generator.class_indices)
# Lalu copy key-nya ke sini secara berurutan.
GRADING_LABELS = [
    # CONTOH (GANTI DENGAN MILIK ANDA):
    'apple_green_a', 'apple_green_b', 'apple_green_rotten',
    'apple_red_a', 'apple_red_b','apple_red_c','apple_red_rotten',
    'banana_a', 'banana_b', 'banana_c', 'banana_rotten', 'durian_grade_a', 'durian_grade_b', 'durian_grade_c',
    'durian_rotten', 'grape_green_a', 'grape_green_b', 'grape_purple_a', 'grape_purple_b' 'grape_rotten', 'guava_grade_a', 'guava_grade_b', 'guava_rotten',
    'jackfruit_a', 'jackfruit_b', 'jackfruit_c', 'jackfruit_rotten', 'mango_grade_a', 'mango_grade_b', 'mango_grade_c', 'mango_rotten', 'orange_grade_a', 'orange_grade_b', 'orange_rotten',
    'papaya_grade_a', 'papaya_grade_b', 'papaya_grade_c', 'papaya_rotten', 'pineapple_grade_a', 'pineapple_grade_b', 'pineapple_rotten',
    'watermelon_grade_a', 'watermelon_grade_b', 'watermelon_rotten'
]

# --- 4. Fungsi Preprocessing Gambar ---
def preprocess_image(image_bytes):
    # Buka gambar dari bytes dan convert ke RGB (hilangkan alpha channel jika png)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize gambar ke ukuran yang diharapkan model (224x224)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Ubah ke numpy array
    image_array = np.array(image, dtype=np.float32)
    
    # Normalisasi nilai pixel ke range 0-1 (karena kita pakai rescale=1./255 di Colab)
    image_array = image_array / 255.0
    
    # Tambahkan dimensi batch (Menjadi: 1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# --- 5. Setup Server Flask ---
app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Cek apakah ada file gambar yang dikirim
    if 'image' not in request.files:
        return jsonify({'error': 'File gambar (key: image) tidak ditemukan'}), 400

    file = request.files['image']
    image_bytes = file.read()

    try:
        # Preprocess gambar
        processed_image = preprocess_image(image_bytes)

        # --- JALANKAN MODEL 1: IDENTIFIKASI ---
        interpreter_identifier.set_tensor(id_input_details[0]['index'], processed_image)
        interpreter_identifier.invoke()
        
        id_output_data = interpreter_identifier.get_tensor(id_output_details[0]['index'])
        id_class_id = np.argmax(id_output_data[0]) # Ambil index dengan probabilitas tertinggi
        predicted_fruit_name = IDENTIFIER_LABELS[id_class_id]
        
        
        # --- JALANKAN MODEL 2: GRADING ---
        interpreter_grading.set_tensor(grade_input_details[0]['index'], processed_image)
        interpreter_grading.invoke()
        
        grade_output_data = interpreter_grading.get_tensor(grade_output_details[0]['index'])
        grade_class_id = np.argmax(grade_output_data[0]) # Ambil index dengan probabilitas tertinggi
        predicted_grade_label = GRADING_LABELS[grade_class_id]

        # --- Return Hasil JSON ---
        print(f"Prediksi: {predicted_fruit_name} | {predicted_grade_label}")
        
        return jsonify({
            'predicted_fruit_name': predicted_fruit_name,
            'predicted_grade_label': predicted_grade_label
        })

    except Exception as e:
        print(f"Error saat inferensi: {e}")
        return jsonify({'error': f'Gagal memproses gambar: {str(e)}'}), 500
    
# --- 6. Jalankan Server ---
if __name__ == '__main__':
    print("🚀 Server ML Python berjalan di http://localhost:5001")
    # debug=True berguna untuk melihat error detail saat development
    app.run(port=5001, debug=True)