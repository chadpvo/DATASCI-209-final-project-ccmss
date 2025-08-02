import sys
import os
import pandas as pd
from multi_processor import MultiSensorDataProcessor
from random import choice

# Set base path for all scenarios
SCENARIO_BASE_PATH = os.path.abspath("data/src/icmcis-drone-detection/train/train")
available_scenarios = [d for d in os.listdir(SCENARIO_BASE_PATH) if os.path.isdir(os.path.join(SCENARIO_BASE_PATH, d))]
available_scenarios.sort()

if not available_scenarios:
    print("No scenarios found in data/src/icmcis-drone-detection/train/train", file=sys.stderr)
    sys.exit(1)

# Load first scenario by default
# current_scenario = available_scenarios[0]

# Load first scenario by default using a mutable container
state = {
    "current_scenario": available_scenarios[0]
}


try:
    #scenario_path = os.path.join(SCENARIO_BASE_PATH, current_scenario)
    scenario_path = os.path.join(SCENARIO_BASE_PATH, state["current_scenario"])
    processor = MultiSensorDataProcessor(scenario_path)
    if not processor.load_all_data():
        raise RuntimeError("Loading sensor data failed.")
    merged_data = processor.merge_sensor_data()
    if merged_data is None:
        raise RuntimeError("Merging sensor data failed.")
except Exception as e:
    print(f"Startup error: {e}", file=sys.stderr)
    sys.exit(1)


def create_flask_app(merged_data):
    from flask import Flask, render_template, jsonify, request
    import json
    import traceback


    app = Flask(__name__)
    display_data = merged_data.copy()
    display_data['datetime(utc)'] = display_data['datetime(utc)'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    def compute_bounds(df):
        buffer = 0.001
        return {
            'min_latitude': float(df['gt_latitude'].min()) - buffer,
            'max_latitude': float(df['gt_latitude'].max()) + buffer,
            'min_longitude': float(df['gt_longitude'].min()) - buffer,
            'max_longitude': float(df['gt_longitude'].max()) + buffer,
            'min_altitude': float(df['gt_altitude'].min()),
            'max_altitude': float(df['gt_altitude'].max()),
            'min_time': df['datetime(utc)'].min(),
            'max_time': df['datetime(utc)'].max(),
            'total_frames': len(df)
        }

    bounds = compute_bounds(display_data)

    @app.route('/')
    def dashboard():
        images = ['bg.png']
        chosen_image = choice(images)
        drone_data_json = json.dumps(display_data.to_dict(orient='records'))
        return render_template(
            'fusion_dashboard.html',
            drone_data_json=drone_data_json,
            background_image=chosen_image,
            current_scenario=state["current_scenario"],
            #current_scenario=current_scenario,
            scenarios=available_scenarios,
            **bounds
        )

    @app.route('/api/switch_scenario', methods=['POST'])
    def switch_scenario():
        #nonlocal display_data, bounds, current_scenario
        try:
            data = request.get_json()
            new_scenario = data.get('scenario')
            print(f"[DEBUG] Requested switch to scenario: {new_scenario}")
            if new_scenario not in available_scenarios:
                return jsonify({'success': False, 'error': 'Scenario not found'}), 400

            new_path = os.path.join(SCENARIO_BASE_PATH, new_scenario)
            print(f"[DEBUG] Scenario folder path: {new_path}")
            print(f"[DEBUG] Files found in {new_path}: {os.listdir(new_path)}")

            print(f"Switching to scenario: {new_scenario}")
            print(f"Path: {new_path}")

            processor = MultiSensorDataProcessor(new_path)
            if not processor.load_all_data():
                return jsonify({'success': False, 'error': 'Failed to load data'}), 500
            new_data = processor.merge_sensor_data()
            if new_data is None:
                return jsonify({'success': False, 'error': 'Failed to merge data'}), 500

            new_data['datetime(utc)'] = new_data['datetime(utc)'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
            display_data = new_data
            bounds = compute_bounds(display_data)
            # current_scenario = new_scenario
            state["current_scenario"] = new_scenario
            return jsonify({
            'success': True,
            'message': f'Switched to {new_scenario}',
            'bounds': bounds,
            'drone_data_json': display_data.to_dict(orient='records') 
        })
        except Exception as e:
            #return jsonify({'success': False, 'error': str(e)}), 500
            print("[ERROR] Exception in switch_scenario:")
            traceback.print_exc()  # This logs full stack trace to terminal
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/performance_metrics')
    def get_performance_metrics():
        try:
            metrics = {}
            if 'alvira_pos_error_m' in display_data.columns:
                alvira_valid = display_data.dropna(subset=['alvira_pos_error_m'])
                if len(alvira_valid) > 0:
                    metrics['alvira'] = {
                        'detections': len(alvira_valid),
                        'detection_rate': len(alvira_valid) / len(display_data) * 100,
                        'mean_pos_error': float(alvira_valid['alvira_pos_error_m'].mean()),
                        'max_pos_error': float(alvira_valid['alvira_pos_error_m'].max()),
                        'mean_alt_error': float(alvira_valid['alvira_alt_error_m'].mean()) if 'alvira_alt_error_m' in alvira_valid.columns else 0
                    }

            if 'arcus_pos_error_m' in display_data.columns:
                arcus_valid = display_data.dropna(subset=['arcus_pos_error_m'])
                if len(arcus_valid) > 0:
                    metrics['arcus'] = {
                        'detections': len(arcus_valid),
                        'detection_rate': len(arcus_valid) / len(display_data) * 100,
                        'mean_pos_error': float(arcus_valid['arcus_pos_error_m'].mean()),
                        'max_pos_error': float(arcus_valid['arcus_pos_error_m'].max()),
                        'mean_alt_error': float(arcus_valid['arcus_alt_error_m'].mean()) if 'arcus_alt_error_m' in arcus_valid.columns else 0
                    }

            if 'diana_snr' in display_data.columns:
                diana_valid = display_data.dropna(subset=['diana_snr'])
                if len(diana_valid) > 0:
                    metrics['diana'] = {
                        'detections': len(diana_valid),
                        'detection_rate': len(diana_valid) / len(display_data) * 100,
                        'mean_snr': float(diana_valid['diana_snr'].mean()),
                        'max_range': float(diana_valid['diana_range'].max()) if 'diana_range' in diana_valid.columns else 0
                    }

            if 'venus_frequency' in display_data.columns:
                venus_valid = display_data.dropna(subset=['venus_frequency'])
                if len(venus_valid) > 0:
                    metrics['venus'] = {
                        'detections': len(venus_valid),
                        'detection_rate': len(venus_valid) / len(display_data) * 100,
                        'mean_frequency': float(venus_valid['venus_frequency'].mean())
                    }

            return jsonify(metrics)
        except Exception as e:
            return jsonify({"error": f"Error calculating metrics: {e}"}), 500

    @app.route('/api/status')
    def get_status():
        return jsonify({
            "status": "active",
            #"current_scenario": current_scenario,
            "current_scenario": state["current_scenario"],
            "total_records": len(display_data),
            "sensors_active": len([col for col in display_data.columns if any(sensor in col for sensor in ['alvira', 'arcus', 'diana', 'venus'])]),
            "time_range": f"{bounds['min_time']} to {bounds['max_time']}"
        })

    return app


app = create_flask_app(merged_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)