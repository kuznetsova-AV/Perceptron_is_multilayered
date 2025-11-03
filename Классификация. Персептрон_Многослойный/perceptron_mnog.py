import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Генерация данных XOR
def generate_xor_data(n_points_per_class=40, noise=0.15):
    np.random.seed(42)
    centers = [
        (0, 0),  # класс 0
        (0, 1),  # класс 1
        (1, 0),  # класс 1
        (1, 1)   # класс 0
    ]
    labels = [0, 1, 1, 0]
    X, y = [], []
    for (cx, cy), label in zip(centers, labels):
        X.append(np.random.normal(loc=(cx, cy), scale=noise, size=(n_points_per_class, 2)))
        y += [label] * n_points_per_class
    return np.vstack(X), np.array(y)

# Обучение сети
X, y = generate_xor_data()

model = MLPClassifier(hidden_layer_sizes=(2,), activation='tanh',
                      solver='adam', max_iter=5000, random_state=1)
model.fit(X, y)

# Проверка XOR
print("\nПроверка логики XOR:")
for a, b in [(0,0), (0,1), (1,0), (1,1)]:
    pred = model.predict([[a,b]])[0]
    print(f"({a}, {b}) -> {pred}")

# Извлекаем веса
W1, b1 = model.coefs_[0], model.intercepts_[0]
W2, b2 = model.coefs_[1], model.intercepts_[1]

# Сетка для визуализации
xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 400), np.linspace(-0.3, 1.3, 400))
grid = np.c_[xx.ravel(), yy.ravel()]
hidden_output = np.tanh(np.dot(grid, W1) + b1)
final_output = model.predict(grid).reshape(xx.shape)

#3 графика: OR / NAND / XOR
fig, axes = plt.subplots(1, 3, figsize=(18,6))

axes[0].contourf(xx, yy, hidden_output[:,0].reshape(xx.shape), cmap='coolwarm', alpha=0.6)
axes[0].scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k', s=50)
axes[0].set_title("1-й нейрон скрытого слоя (примерно OR)")
axes[0].set_xlabel("x₁"); axes[0].set_ylabel("x₂"); axes[0].grid(True)

axes[1].contourf(xx, yy, hidden_output[:,1].reshape(xx.shape), cmap='coolwarm', alpha=0.6)
axes[1].scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k', s=50)
axes[1].set_title("2-й нейрон скрытого слоя (примерно NAND)")
axes[1].set_xlabel("x₁"); axes[1].set_ylabel("x₂"); axes[1].grid(True)

axes[2].contourf(xx, yy, final_output, cmap='coolwarm', alpha=0.35)
axes[2].scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k', s=50)
axes[2].set_title("Выход сети — XOR (диагональное разделение классов)")
axes[2].set_xlabel("x₁"); axes[2].set_ylabel("x₂"); axes[2].grid(True)

plt.suptitle("Визуализация логики многослойного персептрона (XOR)", fontsize=15)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

# 4-й график: Финальный XOR крупно с подписями классов
plt.figure(figsize=(7,7))
plt.contourf(xx, yy, final_output, cmap='coolwarm', alpha=0.35)
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k', s=70)
plt.title("Многослойный персептрон — функция XOR\n(Диагональное разделение классов)", fontsize=13)
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.grid(True)

# Подписи классов
plt.text(0.15, 0.15, "0", fontsize=16, weight='bold', color='black')
plt.text(0.8, 0.15, "1", fontsize=16, weight='bold', color='black')
plt.text(0.15, 0.85, "1", fontsize=16, weight='bold', color='black')
plt.text(0.8, 0.85, "0", fontsize=16, weight='bold', color='black')

plt.show()
