from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'model', 'kmeans_model.pkl')
data_path = os.path.join(BASE_DIR, 'data', 'Customer_Data.csv')

data_exported = joblib.load(model_path)
model = data_exported['model']
scaler = data_exported['scaler']
df = pd.read_csv(data_path).dropna()

def get_plot():
    X = df[['Annual_Income', 'Spending_Score']]
    X_scaled = scaler.transform(X)
    df['Cluster'] = model.predict(X_scaled)
    
    plt.style.use('dark_background')
    
    plt.figure(figsize=(10, 5))
    plt.scatter(df['Annual_Income'], df['Spending_Score'], c=df['Cluster'], cmap='viridis', edgecolors='white', s=50)
    plt.title('Распределение клиентов (Карта кластеров)')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', facecolor='#1e1e1e')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

@app.route('/', methods=['GET', 'POST'])
def index():
    chart2 = get_plot()
    table_data = df.head(15).to_dict(orient='records')
    result = None
    
    if request.method == 'POST':
        try:
            income = float(request.form.get('income', 0))
            score = float(request.form.get('score', 0))
            scaled = scaler.transform([[income, score]])
            cluster_id = model.predict(scaled)[0]
            segments = {0: "Экономный", 1: "Активный клиент", 2: "Целевой", 3: "Транжира", 4: "Средний"}
            result = segments.get(cluster_id, f"Кластер {cluster_id}")
        except:
            result = "Ошибка ввода"

    return render_template('index.html', chart2=chart2, result=result, table_data=table_data)

if __name__ == '__main__':
    app.run(debug=True) 
    
