import joblib
import pandas as pd
import os

# Определяем пути (так же, как в train.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'kmeans_model.pkl')
data_path = os.path.join(BASE_DIR, 'data', 'Customer_Data.csv')

# 1. Загружаем модель, скалер и данные
if not os.path.exists(model_path):
    print("Ошибка: Файл модели не найден! Сначала запусти train.py")
    exit()

data_exported = joblib.load(model_path)
model = data_exported['model']
scaler = data_exported['scaler']

df = pd.read_csv(data_path)

def predict_customer_cluster():
    print("-" * 30)
    print("ТЕСТИРОВАНИЕ КЛАСТЕРИЗАЦИИ")
    print("-" * 30)
    
    # Предлагаем выбрать: ввести вручную или взять из базы
    choice = input("Выбрать случайного клиента из базы (1) или ввести вручную (2)? ")

    if choice == '1':
        # Берем случайную строку из CSV
        random_client = df.sample(1).iloc[0]
        age = random_client['Age']
        income = random_client['Annual_Income']
        score = random_client['Spending_Score']
        gender = random_client['Gender']
        
        print(f"\nДанные клиента из базы:")
        print(f"Пол: {gender}, Возраст: {age}")
        print(f"Доход: ${income}, Баллы трат: {score}")
    else:
        # Ручной ввод
        try:
            income = float(input("Введите годовой доход (Annual_Income): "))
            score = float(input("Введите баллы трат (Spending_Score 1-100): "))
            print(f"\nПроверка для введенных данных: Доход {income}, Траты {score}")
        except ValueError:
            print("Ошибка: вводите только числа!")
            return

    # 2. МАСШТАБИРОВАНИЕ (Важнейший этап!)
    # Модель обучалась на нормализованных данных, поэтому новые данные тоже нужно прогнать через scaler
    new_data = pd.DataFrame([[income, score]], columns=['Annual_Income', 'Spending_Score'])
    new_data_scaled = scaler.transform(new_data)

    
    cluster = model.predict(new_data_scaled)[0]

    print("-" * 30)
    print(f"РЕЗУЛЬТАТ: Клиент относится к КЛАСТЕРУ №{cluster}")
    print("-" * 30)

if __name__ == "__main__":
    predict_customer_cluster()
