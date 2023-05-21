import os
from flask import Flask, jsonify
import mysql.connector
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Database connection
connection = mysql.connector.connect(
    host=os.getenv("HOST"),
    user='admin',
    password=os.getenv("PASSWORD"),
    database='weather'
)

@app.route('/')
def home():
    return 'Welcome to the SkyScan Weather API!'

@app.route('/count')
def get_count():
    try:
        cursor = connection.cursor()
        cursor.execute('SELECT COUNT(*) AS count FROM weather;')
        result = cursor.fetchone()
        count = result[0]
        return jsonify(count=count)
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run()
