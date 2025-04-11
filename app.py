from flask import Flask, request, jsonify, render_template, send_from_directory
import requests
import os
import logging
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Проверка токена
HF_TOKEN = os.getenv("HF_TOKEN")
logger.info(f"API Key loaded: {'Yes' if HF_TOKEN else 'No'}")
if HF_TOKEN:
    logger.info(f"API Key value first 10 chars: {HF_TOKEN[:10]}")

app = Flask(__name__, static_folder='static')

# Конфигурация Hugging Face API
HF_API_URL = "https://api-inference.huggingface.co/models/gpt2"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/generate', methods=['POST'])
def generate_test():
    try:
        # Получаем данные из запроса
        if request.is_json:
            data = request.json
        else:
            data = request.form

        topic = data.get('topic', '')
        difficulty = data.get('difficulty', 'средний')
        count = data.get('count', 5)
        
        # Проверяем загруженный файл
        file = request.files.get('file')
        if file:
            # Читаем содержимое файла
            file_content = file.read().decode('utf-8')
            topic = f"{topic}\n\nКонтекст из файла:\n{file_content}"

        if not topic:
            return jsonify({"error": "Не указана тема теста и не загружен файл"}), 400

        # Формируем промпт
        prompt = f"""
        Сгенерируй тест на тему: {topic}
        Уровень сложности: {difficulty}
        Количество вопросов: {count}
        Формат каждого вопроса:
        1. Вопрос...
        A) Вариант 1
        B) Вариант 2
        C) Вариант 3
        D) Вариант 4
        Правильный ответ: A
        """

        # Отправляем запрос к Hugging Face API
        response = requests.post(
            HF_API_URL,
            headers=HEADERS,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_length": 1000,
                    "temperature": 0.7
                }
            }
        )

        # Обрабатываем ответ
        if response.status_code != 200:
            error_msg = response.json().get("error", "Неизвестная ошибка API")
            logger.error(f"Hugging Face API error: {error_msg}")
            return jsonify({
                "error": f"Ошибка Hugging Face API: {error_msg}",
                "status_code": response.status_code
            }), 500

        generated_text = response.json()[0]['generated_text']
        return jsonify({"test": generated_text})

    except Exception as e:
        logger.error(f"Error in generate_test: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Получаем порт из переменной окружения или используем 8080 по умолчанию
    port = int(os.environ.get('PORT', 8080))
    
    # Запускаем приложение на всех интерфейсах
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False  # Отключаем debug в production
    )