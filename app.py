"""
Flask backend for the fire-aqi-map app.

Serves NASA FIRMS fire detections as JSON so the frontend map can plot them,
and serves the static map page itself.
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from fires import get_fires, FireDataError

app = Flask(__name__, static_folder="static")
CORS(app)


@app.route("/api/fires")
def fires_endpoint():
    """Return recent India fire detections as JSON for the map."""
    try:
        df = get_fires()
    except FireDataError as e:
        return jsonify({"error": str(e)}), 502

    records = df[["latitude", "longitude", "acq_date", "confidence", "frp"]]
    return jsonify(records.to_dict(orient="records"))


@app.route("/")
def index():
    """Serve the map page."""
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
