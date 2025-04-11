from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
import os
from dotenv import load_dotenv
import logging
import socket
import traceback

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__, static_folder='static')
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_API_KEY = os.getenv('HF_API_KEY')

# Логируем информацию о ключе (без самого ключа)
logger.info(f"API Key loaded: {'Yes' if HF_API_KEY else 'No'}")
logger.info(f"API Key value first 10 chars: {HF_API_KEY[:10] if HF_API_KEY else 'None'}")

# Получаем IP-адрес хоста
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
logger.info(f"Hostname: {hostname}")
logger.info(f"IP Address: {ip_address}")

@app.route('/')
def index():
    logger.debug("Index route accessed")
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    logger.debug(f"Serving static file: {filename}")
    return send_from_directory(app.static_folder, filename)

def generate_with_huggingface(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }
        logger.debug(f"Sending request to {HF_API_URL}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Payload: {payload}")
        
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response content: {response.text}")
        
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"API Error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return {"error": error_msg}
    except Exception as e:
        error_msg = f"Exception in generate_with_huggingface: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"error": error_msg}

@app.route('/generate', methods=['POST'])
def generate_test():
    try:
        logger.debug("Generate route accessed")
        logger.debug(f"Request data: {request.data}")
        
        data = request.json
        logger.debug(f"Parsed JSON data: {data}")
        
        topic = data['topic']
        difficulty = data['difficulty']
        count = data['count']

        prompt = f"""Сгенерируй тест на тему {topic}. 
        Сложность: {difficulty}. 
        {count} вопросов с 4 вариантами ответов (1 правильный). 
        Формат: '1. Вопрос...\nA) Вариант 1\nB) Вариант 2...'"""

        logger.debug(f"Generated prompt: {prompt}")
        
        response = generate_with_huggingface(prompt)
        logger.debug(f"API response: {response}")
        
        if "error" in response:
            logger.error(f"Error from API: {response['error']}")
            return jsonify({'success': False, 'error': response["error"]})
        
        if isinstance(response, list) and len(response) > 0:
            test_content = response[0]['generated_text']
            return jsonify({'success': True, 'test': test_content})
        else:
            error_msg = 'Неверный формат ответа от API'
            logger.error(error_msg)
            return jsonify({'success': False, 'error': error_msg})
    except Exception as e:
        error_msg = f"Exception in generate_test: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(
        debug=True,
        host='127.0.0.1',
        port=8080,
        use_reloader=True,
        use_debugger=True,
        threaded=True
    )