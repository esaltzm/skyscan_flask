import os
import json
from flask import Flask, jsonify, send_file
from io import BytesIO
import mysql.connector
from flask_cors import CORS
import numpy as np
import rasterio as rio
from scipy.interpolate import griddata
from PIL import Image

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

@app.route('/weather/<param>/<time>/<coords>')
def interpolate_weather_data(param, time, coords):
    cursor = connection.cursor(dictionary=True)
    coords = json.loads(coords)
    lowerLat, upperLat = coords[0][0], coords[1][0]
    lowerLng, upperLng = coords[0][1], coords[1][1]

    query = f"""
        SELECT latitude, longitude, {param}
        FROM weather
        WHERE latitude > {lowerLat}
            AND latitude < {upperLat}
            AND longitude > {lowerLng}
            AND longitude < {upperLng}
            AND time_start = {time};
    """
    cursor.execute(query)
    results = cursor.fetchall()
    temperatureData = results  # Placeholder for temperature data

    width = 700  # Width of the 2D array
    height = 400  # Height of the 2D array
    arr = np.full((height, width), np.nan)

    minTemperature = min(data['t'] for data in temperatureData)
    maxTemperature = max(data['t'] for data in temperatureData)

    for data in temperatureData:
        x = int((float(data['longitude']) - lowerLng) * (width / (upperLng - lowerLng)))
        y = int((upperLat - float(data['latitude'])) * (height / (upperLat - lowerLat)))
        arr[y][x] = float(data['t'])
    
    arr = fill_nans_bilinear(arr)

    # Create a blank PIL image using mode='RGB'
    height, width = arr.shape
    image = Image.new('RGB', (width, height))

    # Iterate over each pixel in the array
    for y in range(height):
        for x in range(width):
            pixel_value = arr[y, x]

            # Check if the pixel value is NaN
            if not np.isnan(pixel_value):
                # Find the nearest index in the color scale based on the pixel value
                index = np.interp(pixel_value, color_scale[:, 0], np.arange(len(color_scale)))
                # Get the neighboring indices for interpolation
                lower_index = int(np.floor(index))
                upper_index = int(np.ceil(index))

                # Get the RGB values for the lower and upper indices
                lower_rgb = color_scale[lower_index, 1:]
                upper_rgb = color_scale[upper_index, 1:]

                # Calculate the interpolation factor
                factor = index - lower_index

                # Interpolate the RGB values
                interpolated_rgb = (1 - factor) * lower_rgb + factor * upper_rgb

                # Set the color of the pixel in the image
                r, g, b = tuple(interpolated_rgb.clip(0, 255).astype(int))
                image.putpixel((x, y), (r, g, b))
            else:
                image.putpixel((x, y), (0, 0, 0))
    # Create an in-memory stream
    img_stream = BytesIO()

    # Save the PIL image to the stream as PNG
    image.save(img_stream, format='PNG')

    # Set the stream position to the beginning
    img_stream.seek(0)

    # Return the image as a response
    return send_file(img_stream, mimetype='image/png')


def fill_nans_bilinear(array):
    x, y = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
    points = np.column_stack((x.flatten(), y.flatten()))
    values = array.flatten()
    valid_indices = ~np.isnan(values)
    filled_values = griddata(points[valid_indices], values[valid_indices], (x, y), method='cubic')
    return filled_values

if __name__ == '__main__':
    app.run()