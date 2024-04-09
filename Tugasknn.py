import numpy as np

x_train = np.array([[40, 5, 60],
                    [50, 8, 40],
                    [50, 7, 30],
                    [70, 4, 60],
                    [80, 4, 80],
                    [60, 6, 60]])

y_train = np.array(['Jelek', 'Bagus', 'Jelek', 'Bagus', 'Bagus', 'Bagus'])
x_test = np.array([50, 3, 40])

def predict_knn(k):
    distances = np.linalg.norm(x_train - x_test, axis=1)
    nearest_indices = np.argsort(distances)[:k]
    nearest_neighbors = y_train[nearest_indices]
    unique, counts = np.unique(nearest_neighbors, return_counts=True)
    prediction = unique[np.argmax(counts)]
    return prediction

print(f"Kelas prediksi untuk data uji dengan K = 3 adalah : '{predict_knn(3)}'.")
print(f"Kelas prediksi untuk data uji dengan K = 4 adalah : '{predict_knn(4)}'.")
print(f"Kelas prediksi untuk data uji dengan K = 5 adalah : '{predict_knn(5)}'.")
