""" Purpose: Provides a web-based dashboard to monitor the real-time activity recognition results. It uses Flask, a lightweight web framework, to create a local web server that displays the latest activity (and potentially position) detected by the HAR system. This dashboard can be accessed from a browser (for example, on a phone or laptop connected to the same network as the Raspberry Pi) to see what the system is detecting in real time. Features:
Launches a Flask app on the device (default http://0.0.0.0:5000 or similar).
The main route (/) displays the current activity and position.
The page auto-refreshes every couple of seconds to update the information, or it could use AJAX/JavaScript for updates (here we use a simple meta refresh for simplicity).
The dashboard reads the experiments/results/current_pred.txt file (which real_time_inference.py updates) to get the latest prediction. This decouples the dashboard from the inference process, allowing them to run in parallel.
Flask is run in debug or production mode as needed. In real deployment, you might run it via a WSGI server or as a Flask app in production mode.
Usage:
bash
Copy
python dashboard.py
Then open a browser to the device’s IP on port 5000 (e.g., http://localhost:5000 if local, or http://<raspberrypi_ip>:5000 from another device). You should see the page showing current activity and position, updating periodically. Ensure that real_time_inference.py is running simultaneously so that the file is being updated."""

# code starts from here 

#!/usr/bin/env python3
"""
dashboard.py: Flask web dashboard for real-time HAR monitoring.

Starts a web server that displays the latest recognized activity and (optionally) the participant's position.
It reads from the file written by real_time_inference.py (experiments/results/current_pred.txt) to get current status.

Usage:
    python dashboard.py
Then open a browser at http://<device_ip>:5000 to view the dashboard.
"""
from flask import Flask, Response
import os
app = Flask(__name__)

DATA_FILE = "experiments/results/current_pred.txt"

@app.route('/')
def index():
    # Read the latest prediction from file
    activity = "Unknown"
    pos_text = ""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                line = f.readline().strip()
                if line:
                    parts = line.split(',')
                    activity = parts[0] if parts[0] else "Unknown"
                    if len(parts) >= 3 and parts[1] and parts[2]:
                        pos_text = f"Position: ({parts[1]}, {parts[2]})"
        except:
            activity = "Unknown"
    # Construct an HTML response
    html_content = f"""
    <html>
      <head>
        <title>HAR Dashboard</title>
        <meta http-equiv="refresh" content="2" />  <!-- refresh page every 2 seconds -->
        <style>
          body {{ font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }}
          h1 {{ font-size: 3em; }}
          p {{ font-size: 1.5em; }}
        </style>
      </head>
      <body>
        <h1>Current Activity: {activity}</h1>
        <p>{pos_text}</p>
        <p><em>Last updated: just now</em></p>
      </body>
    </html>
    """
    return Response(html_content, mimetype='text/html')

if __name__ == "__main__":
    # Run the Flask development server. In production, use a proper WSGI server or Flask's production mode.
    # host='0.0.0.0' makes it accessible externally, port 5000 is default.
    app.run(host='0.0.0.0', port=5000, debug=False)


""" Comments: This simple dashboard will display the latest recognized activity as a large headline, and the position if available. The page auto-refreshes every 2 seconds (via meta tag), which is a simple way to update the info. For a smoother update without full page reload, one could use JavaScript to periodically fetch new data (e.g., an AJAX call to an endpoint that returns JSON). To keep things straightforward, we did a full refresh. Make sure the DATA_FILE path matches what the inference script writes. Here it's experiments/results/current_pred.txt. The file reading is done at each page request. This is efficient enough given the small size (one line) and low frequency of requests. Potential enhancements:
If you want to visualize more, you could show a history of the last few activities or a timeline.
You could integrate a simple chart or icon for each activity (e.g., walking icon, falling icon, etc.) if you have that mapping.
For position, you could embed a simple canvas or image of a room layout and plot the coordinates of the participant, updating as they move. This would require more front-end code (JavaScript) to draw the position.
Add a route for a JSON API (e.g., /latest returning {"activity": "...", "x": ..., "y": ...}) and have the page fetch that via JS. This way you don’t reload the entire page.
The Flask app currently runs in single-threaded mode (unless you specify otherwise). For low update rates this is fine. If you anticipate high load or want asynchronous updates, consider running with a production server or using Flask’s threaded mode.
The dashboard is meant for monitoring and demonstration. It can be extended or styled as needed, but as provided it gives a clear, quickly accessible interface to see the HAR system’s output in real time. """ 
