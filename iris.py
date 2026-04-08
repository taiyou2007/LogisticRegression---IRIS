import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Load the Iris dataset
def loadIrisData():
    dataset = []
    label_map = {
        'Iris-setosa': [1, 0, 0],
        'Iris-versicolor': [0, 1, 0],
        'Iris-virginica': [0, 0, 1]
    }
    with open(r"C:\Users\Admin\Desktop\scikit learn\LogisticRegression---IRIS\iris\iris.data",'r') as file:
        for line in file:
            if line.strip():
                parts = line.strip().split(',')
                features = list(map(float, parts[:-1]))
                label = label_map[parts[-1]]
                dataset.append((features, label))
    random.seed(42)
    random.shuffle(dataset)
    return dataset

# softmax function
def softmax(logits):
    max_logit = max(logits)
    
    exp_values = []
    for value in logits:
        exp_values.append(math.exp(value - max_logit))

    total = sum(exp_values)
    probabilities = []
    for prob in exp_values:
        probabilities.append(prob / total)

    return probabilities

#Loss
def cross_entropy_loss(pred, labels):
    loss = 0.0
    epsilon = 1e-15
    for i in range(len(labels)):
        if labels[i] == 1:
            loss += -labels[i]*math.log(pred[i]+epsilon)
    
    return loss

#y_pred
def predict(features, w, b):
    logits = [0.0, 0.0, 0.0]
    for i in range(3):
        dot_product = 0.0
        for j in range(len(features)):
            dot_product += features[j] * w[j][i]
        logits[i] = dot_product + b[i]
    
    return softmax(logits)

#accuracy
def calculate_accuracy(dataset, W, b):
    correct = 0
    for features, true_label in dataset:
        probs = predict(features, W, b)
        if probs.index(max(probs)) == true_label.index(1):
            correct += 1
    return (correct / len(dataset)) * 100

#train
def train_model_ewma(train_data, test_data=None, num_features=4, epochs=150, alpha=0.2, beta=0.9):
    #weights and bias & velocity
    W = [[random.uniform(-0.1, 0.1) for _ in range(3)] for _ in range(num_features)]
    b = [0.0, 0.0, 0.0]
    v_dW = [[0.0 for _ in range(3)] for _ in range(num_features)]
    v_db = [0.0, 0.0, 0.0]
    
    history = {'loss': [], 'acc': []}
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, true_label in train_data:
            probs = predict(features, W, b)
            epoch_loss += cross_entropy_loss(probs, true_label)
            dz = [probs[i] - true_label[i] for i in range(3)]
            
            for class_idx in range(3):
                #bias update
                db = dz[class_idx]

                # v_db = beta * v_db + (1 - beta) * db
                v_db[class_idx] = (beta * v_db[class_idx]) + ((1 - beta) * db)
                # b = b - alpha * v_db
                b[class_idx] -= alpha * v_db[class_idx]
                
                # weights update
                for f_idx in range(num_features):
                    dW = features[f_idx] * dz[class_idx]

                    # v_dW = beta * v_dW + (1 - beta) * dW
                    v_dW[f_idx][class_idx] = (beta * v_dW[f_idx][class_idx]) + ((1 - beta) * dW)

                    # W = W - alpha * v_dW
                    W[f_idx][class_idx] -= alpha * v_dW[f_idx][class_idx]
                    
        # Lưu kết quả mỗi epoch để sau này vẽ
        history['loss'].append(epoch_loss / len(train_data))
        if test_data is not None:
            history['acc'].append(calculate_accuracy(test_data, W, b))
            
    return W, b, history

print("Đang tải dữ liệu và chia tỷ lệ 80/20...")
dataset = loadIrisData()
split_idx = int(len(dataset) * 0.8)
train_set = dataset[:split_idx]
test_set = dataset[split_idx:]

print("Đang huấn luyện mô hình 4 Đặc trưng (Full)...")
W_full, b_full, history_full = train_model_ewma(
    train_set, test_set, num_features=4, epochs=200, alpha=0.2, beta=0.9
)

print("Đang huấn luyện mô hình 2 Đặc trưng (Để vẽ Ranh giới 2D)...")
train_set_2d = [([row[0][2], row[0][3]], row[1]) for row in train_set]
W_2d, b_2d, _ = train_model_ewma(
    train_set_2d, test_data=None, num_features=2, epochs=200, alpha=0.2, beta=0.9
)

# ==========================================
# PHẦN 4: VẼ 5 BIỂU ĐỒ (DASHBOARD) - BẢN RỘNG RÃI
# ==========================================
print("Đang dựng hình đồ họa...")
# Tăng kích thước khung ảnh lên (Rộng 20, Cao 16)
plt.figure(figsize=(20, 16)) 

# --- 1. Loss over Epochs ---
plt.subplot(3, 2, 1) # Đổi thành lưới 3 hàng, 2 cột, vị trí số 1
plt.plot(history_full['loss'], color='#d62728', linewidth=2.5)
plt.title('1. Mức độ Sai sót (Loss over Epochs)', fontsize=14, fontweight='bold')
plt.xlabel('Vòng lặp (Epochs)', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# --- 2. Accuracy on Test Set ---
plt.subplot(3, 2, 2) # Vị trí số 2
plt.plot(history_full['acc'], color='#2ca02c', linewidth=2.5)
plt.title('2. Độ chính xác (Accuracy trên Test Set)', fontsize=14, fontweight='bold')
plt.xlabel('Vòng lặp (Epochs)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# --- 3. Feature Weights ---
plt.subplot(3, 2, 3) # Vị trí số 3
feature_names = ['Đài hoa (Dài)', 'Đài hoa (Rộng)', 'Cánh hoa (Dài)', 'Cánh hoa (Rộng)']
x_pos = np.arange(len(feature_names))
width = 0.25
W_np = np.array(W_full)

plt.bar(x_pos - width, W_np[:, 0], width, label='Setosa', color='#FFaaaa')
plt.bar(x_pos, W_np[:, 1], width, label='Versicolor', color='#aaFFaa')
plt.bar(x_pos + width, W_np[:, 2], width, label='Virginica', color='#aaaaFF')
plt.xticks(x_pos, feature_names, rotation=15, fontsize=11)
plt.title('3. Tầm quan trọng của Đặc trưng (Weights)', fontsize=14, fontweight='bold')
plt.axhline(0, color='black', linewidth=1)
plt.legend(fontsize=11)

# --- 4. Decision Boundary (Dùng mô hình 2 Đặc trưng) ---
plt.subplot(3, 2, 4) # Vị trí số 4
X_pts = [row[0][0] for row in train_set_2d]
Y_pts = [row[0][1] for row in train_set_2d]
Labels = [row[1].index(1) for row in train_set_2d] 

x_min, x_max = min(X_pts) - 0.5, max(X_pts) + 0.5
y_min, y_max = min(Y_pts) - 0.5, max(Y_pts) + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = []
for x_val, y_val in zip(xx.ravel(), yy.ravel()):
    probs = predict([x_val, y_val], W_2d, b_2d)
    Z.append(probs.index(max(probs)))
Z = np.array(Z).reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
plt.scatter(X_pts, Y_pts, c=Labels, cmap=cmap_bold, edgecolors='k', s=50)
plt.title('4. Ranh giới quyết định (Petal Length & Width)', fontsize=14, fontweight='bold')
plt.xlabel('Chiều dài cánh hoa', fontsize=12)
plt.ylabel('Chiều rộng cánh hoa', fontsize=12)

# --- 5. Confusion Matrix ---
plt.subplot(3, 2, 5) # Vị trí số 5 (Góc dưới bên trái)
cm = np.zeros((3, 3), dtype=int)
for features, true_label in test_set:
    probs = predict(features, W_full, b_full)
    pred_class = probs.index(max(probs))
    true_class = true_label.index(1)
    cm[true_class][pred_class] += 1

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14},
            xticklabels=['Setosa', 'Versicolor', 'Virginica'], 
            yticklabels=['Setosa', 'Versicolor', 'Virginica'])
plt.title('5. Ma trận nhầm lẫn (Confusion Matrix)', fontsize=14, fontweight='bold')
plt.xlabel('AI Dự đoán', fontsize=12)
plt.ylabel('Thực tế', fontsize=12)

# Tắt ô số 6 để trống góc dưới cùng bên phải cho thoáng
plt.subplot(3, 2, 6).axis('off')

# Lệnh này cực kỳ quan trọng: pad=3.0 giúp đẩy các đồ thị ra xa nhau!
plt.tight_layout(pad=3.0) 
plt.show()
print("\nHoàn tất!")


