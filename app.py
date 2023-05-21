import os
import json
from flask import Flask, jsonify, send_file
from io import BytesIO
import mysql.connector
from flask_cors import CORS
import numpy as np
import rasterio as rio
from scipy.interpolate import griddata
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from collections import defaultdict
import time as t
import copy

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
    
@app.route('/size')
def get_table_size():
    cursor = connection.cursor(dictionary=True)
    query = """
        SELECT 
            table_name AS `weather`, 
            round(((data_length + index_length) / 1024 / 1024 / 1000), 2) AS `Size in GB`
        FROM information_schema.TABLES
        WHERE table_schema = 'weather'
            AND table_name = 'weather';
    """
    cursor.execute(query)
    results = cursor.fetchall()
    return results

@app.route('/times')
def get_time_range():
    cursor = connection.cursor(dictionary=True)
    query = "SELECT MIN(time_start) AS lowest, MAX(time_start) AS highest FROM weather;"
    cursor.execute(query)
    results = cursor.fetchall()
    return jsonify(results)

@app.route('/minmax/<time>/<param>')
def get_min_max(time, param):
    cursor = connection.cursor(dictionary=True)
    query = f"SELECT MIN({param}) AS min, MAX({param}) AS max FROM weather WHERE time_start = '{time}';"
    cursor.execute(query)
    results = cursor.fetchall()
    return jsonify(results)

color_scale = np.array([[-23.33333333, 255., 180., 255.],
                  [-16.66666667, 255., 0., 255.],
                  [-10., 180., 0., 255.],
                  [-5.33333333, 100., 0., 255.],
                  [0., 0., 0., 255.],
                  [4.66666667, 0., 255., 255.],
                  [10., 255., 255., 255.],
                  [16.66666667, 255., 255., 0.],
                  [23.33333333, 255., 180., 0.],
                  [30., 255., 0., 0.],
                  [36.66666667, 255., 0., 255.],
                  [43.33333333, 255., 255., 255.]])

cache = defaultdict(list)
img_cache = defaultdict(BytesIO)

@app.route('/weather/<param>/<time>/<coords>')
def interpolate_weather_data(param, time, coords):
    cursor = connection.cursor(dictionary=True)
    coords = json.loads(coords)
    lowerLat, upperLat = coords[0][0], coords[1][0]
    lowerLng, upperLng = coords[0][1], coords[1][1]

    query = f"""
        SELECT latitude, longitude, {param}
        FROM weather
        WHERE time_start = {time};
    """

    start_time = t.time()
    if not f'{param}_{time}' in img_cache:
        cursor.execute(query)
        data = cursor.fetchall()
        width = 700  # Width of the 2D array
        height = 400  # Height of the 2D array
        arr = np.full((height, width), np.nan)

        for data in data:
            if data['longitude'] > lowerLng and data['longitude'] < upperLng and data['latitude'] > lowerLat and data['latitude'] < upperLat:
                x = int((float(data['longitude']) - lowerLng) * (width / (upperLng - lowerLng)))
                y = int((upperLat - float(data['latitude'])) * (height / (upperLat - lowerLat)))
                arr[y][x] = float(data['t'])
        
        start_time = t.time()
        print(np.sum(np.isnan(arr)))
        arr = interpolate(arr)
        print(np.sum(np.isnan(arr)))
        end_time = t.time()
        execution_time = end_time - start_time

        print(f"time to interpolate: {execution_time} seconds")

        height, width = arr.shape
        image = Image.new('RGB', (width, height))

        start_time = t.time()

        # Assign variables outside the loop
        pixel_values = arr.flatten()
        color_scale_values = color_scale[:, 0]
        color_scale_rgb = color_scale[:, 1:]

        # Create index array outside the loop
        indices = np.interp(pixel_values, color_scale_values, np.arange(len(color_scale)))

        for y in range(height):
            for x in range(width):
                pixel_value = pixel_values[y * width + x]
                if not np.isnan(pixel_value):
                    index = indices[y * width + x]
                    lower_index = int(np.floor(index))
                    upper_index = int(np.ceil(index))
                    lower_rgb = color_scale_rgb[lower_index]
                    upper_rgb = color_scale_rgb[upper_index]
                    factor = index - lower_index
                    interpolated_rgb = (1 - factor) * lower_rgb + factor * upper_rgb
                    r, g, b = tuple(interpolated_rgb.astype(int))
                    image.putpixel((x, y), (r, g, b))
                else:
                    image.putpixel((x, y), (0, 0, 0))
        end_time = t.time()
        execution_time = end_time - start_time

        print(f"time to color png: {execution_time} seconds")

        img_stream = BytesIO()
        image.save(img_stream, format='PNG')
        img_stream_copy = BytesIO(img_stream.getvalue())

        img_cache[f'{param}_{time}'] = img_stream_copy
        img_stream.seek(0)
        return send_file(img_stream, mimetype='image/png')
    else:
        img_stream = copy.copy(img_cache[f'{param}_{time}'])
        img_stream.seek(0)
        return send_file(img_stream, mimetype='image/png')

def interpolate(array):
    x, y = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
    points = np.column_stack((x.flatten(), y.flatten()))
    values = array.flatten()
    valid_indices = ~np.isnan(values)
    filled_values = griddata(points[valid_indices], values[valid_indices], (x, y), method='linear')
    return filled_values

if __name__ == '__main__':
    app.run()