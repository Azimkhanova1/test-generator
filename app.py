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
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"  # Более мощная модель
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
            try:
                # Читаем содержимое файла
                file_content = file.read().decode('utf-8')
                topic = f"{topic}\n\nКонтекст из файла:\n{file_content}"
            except Exception as e:
                logger.error(f"Error reading file: {str(e)}")
                return jsonify({"error": "Ошибка при чтении файла"}), 400

        if not topic:
            return jsonify({"error": "Не указана тема теста и не загружен файл"}), 400

        # Формируем промпт
        prompt = f"""
        <s>[INST] Сгенерируй тест на тему: {topic}
        Уровень сложности: {difficulty}
        Количество вопросов: {count}
        Формат каждого вопроса:
        1. Вопрос...
        A) Вариант 1
        B) Вариант 2
        C) Вариант 3
        D) Вариант 4
        Правильный ответ: A
        [/INST]
        """

        # Отправляем запрос к Hugging Face API
        try:
            response = requests.post(
                HF_API_URL,
                headers=HEADERS,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_length": 1000,
                        "temperature": 0.7,
                        "return_full_text": False
                    }
                },
                timeout=30  # Увеличиваем таймаут
            )
        except requests.exceptions.Timeout:
            logger.error("Hugging Face API request timed out")
            return jsonify({"error": "Превышено время ожидания ответа от API"}), 504
        except requests.exceptions.RequestException as e:
            logger.error(f"Hugging Face API request failed: {str(e)}")
            return jsonify({"error": f"Ошибка при запросе к API: {str(e)}"}), 502

        # Обрабатываем ответ
        if response.status_code != 200:
            try:
                error_msg = response.json().get("error", "Неизвестная ошибка API")
            except:
                error_msg = f"HTTP {response.status_code}: {response.text}"
            logger.error(f"Hugging Face API error: {error_msg}")
            return jsonify({
                "error": f"Ошибка Hugging Face API: {error_msg}",
                "status_code": response.status_code
            }), 500

        try:
            generated_text = response.json()[0]['generated_text']
            return jsonify({"test": generated_text})
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing API response: {str(e)}")
            return jsonify({"error": "Ошибка при обработке ответа API"}), 500

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