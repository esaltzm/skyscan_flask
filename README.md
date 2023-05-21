# skyscan_flask
Flask version of my weather visualization Node.js backend (implementing server side raster rendering)

This aims to produce higher-resolution visualizations than the first implementation of the SkyScan app - initially, I just queried the database on the backend and sent an array to the frontend to interpolate and color-scale with Plotly.js, but this method was only able to handle <500 data points before crashing the browser.

My vision is for this new backend to use more efficient image processing libraries in python and send a pre-interpolated and colored png raster to the front end.


<img width="715" alt="temperature raster accessed thru postman" src="https://github.com/esaltzm/skyscan_flask/assets/99096893/b8509f7a-56fd-4132-94d5-a55a881e2935">
