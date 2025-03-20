import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load mô hình từ TensorFlow Hub
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = hub.load(model_url)
print("✅ Model loaded successfully!")

# URL ảnh (Bạn có thể thay đổi URL ảnh khác)
image_url = "https://www.basenji.org/BasenjiU/Breeder/TheStandard/ImagesDissect/ColorBrindle.jpg"

# Tải ảnh và chuyển đổi thành định dạng phù hợp
image_path = tf.keras.utils.get_file("image1.jpg", image_url)
image = Image.open(image_path).convert("RGB").resize((224, 224))

# Hiển thị ảnh
plt.imshow(image)
plt.axis("off")
plt.title("Input Image")
plt.show()

# Xử lý ảnh cho mô hình (chuẩn hóa về [0,1] và thêm chiều batch)
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize ảnh về 224x224
    image = np.array(image)  # Chuyển ảnh về numpy array
    if image.shape[-1] == 4:  # Nếu ảnh có 4 kênh (RGBA), chuyển về RGB
        image = image[..., :3]
    image = image / 255.0  # Chuẩn hóa pixel về khoảng [0,1]
    image = np.expand_dims(image, axis=0)  # Thêm batch dimension
    return image.astype(np.float32)  # Đảm bảo kiểu dữ liệu đúng

processed_image = preprocess_image(image)

# Dự đoán với mô hình
predictions = model(processed_image)
predicted_class = np.argmax(predictions, axis=-1)[0]  # Lấy index lớp có xác suất cao nhất
print("Predicted class index:", predicted_class)

# Tải danh sách nhãn từ ImageNet
labels_path = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
labels_file = tf.keras.utils.get_file("ImageNetLabels.txt", labels_path)

# Đọc danh sách nhãn
with open(labels_file, "r") as f:
    labels = f.read().splitlines()

# Kiểm tra số lượng nhãn có đủ hay không
if predicted_class < len(labels):
    predicted_label = labels[predicted_class]
else:
    predicted_label = "Unknown"

print(f"✅ Prediction: {predicted_label}")

# Hiển thị ảnh với nhãn dự đoán
plt.imshow(image)
plt.axis("off")
plt.title(f"Prediction: {predicted_label}")
plt.show()
