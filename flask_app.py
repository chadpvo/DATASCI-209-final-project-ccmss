import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from multi_processor import load_scenario_data, clean_data_for_json

app = Flask(__name__)

# Global variables
current_data = None
available_scenarios = []
current_scenario = None

@app.route('/')
def index():
    global current_data, available_scenarios, current_scenario
    
    # Find scenarios in both possible directory names
    possible_dirs = ['data/src/icmcis-drone-detection/train/train', 'Data/src/icmcis-drone-detection/train/train']
    drone_data_path = None
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            drone_data_path = dir_path
            break
    
    if drone_data_path:
        available_scenarios = [d for d in os.listdir(drone_data_path) 
                             if os.path.isdir(os.path.join(drone_data_path, d))]
        available_scenarios.sort()
    else:
        available_scenarios = []
    
    if not available_scenarios:
        return "<h1>No scenarios found data/src/icmcis-drone-detection/train/train</h1>"
    
    # Load first scenario
    current_scenario = available_scenarios[0]
    current_data = load_scenario_data(current_scenario)
    
    if current_data is None:
        # Try next scenario if first one fails
        for scenario in available_scenarios[1:]:
            current_data = load_scenario_data(scenario)
            if current_data is not None:
                current_scenario = scenario
                break
    
    if current_data is None:
        return f"<h1>Error loading any scenario data</h1>"
    
    # Calculate bounds
    min_lat = float(current_data['gt_latitude'].min() - 0.001)
    max_lat = float(current_data['gt_latitude'].max() + 0.001)
    min_lon = float(current_data['gt_longitude'].min() - 0.001)
    max_lon = float(current_data['gt_longitude'].max() + 0.001)
    
    # Use the multi_scenario_dashboard.html template
    return render_template('fusion_dashboard.html',
                         scenarios=available_scenarios,
                         current_scenario=current_scenario,
                         drone_data_json=current_data.to_json(orient='records'),
                         total_frames=len(current_data),
                         min_latitude=min_lat,
                         max_latitude=max_lat,
                         min_longitude=min_lon,
                         max_longitude=max_lon)

@app.route('/api/switch_scenario', methods=['POST'])
def switch_scenario():
    global current_data, current_scenario
    
    try:
        data = request.get_json()
        new_scenario = data.get('scenario')
        
        print(f"\n=== Switching to scenario: {new_scenario} ===")
        
        if new_scenario not in available_scenarios:
            return jsonify({'success': False, 'error': f'Scenario not found: {new_scenario}'})
        
        new_data = load_scenario_data(new_scenario)
        
        if new_data is None:
            return jsonify({'success': False, 'error': f'Failed to load scenario: {new_scenario}. Check server console for details.'})
        
        # Check if data is empty
        if len(new_data) == 0:
            return jsonify({'success': False, 'error': f'Scenario {new_scenario} has no data after processing'})
        
        # Check for required columns
        required_cols = ['gt_latitude', 'gt_longitude']
        missing_cols = [col for col in required_cols if col not in new_data.columns]
        if missing_cols:
            return jsonify({'success': False, 'error': f'Missing required columns: {missing_cols}'})
        
        # Update global data
        current_data = new_data
        current_scenario = new_scenario
        
        # Calculate bounds with error checking
        try:
            bounds = {
                'total_frames': len(current_data),
                'min_latitude': float(current_data['gt_latitude'].min() - 0.001),
                'max_latitude': float(current_data['gt_latitude'].max() + 0.001),
                'min_longitude': float(current_data['gt_longitude'].min() - 0.001),
                'max_longitude': float(current_data['gt_longitude'].max() + 0.001)
            }
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error calculating bounds: {str(e)}'})
        
        print(f"Successfully loaded {new_scenario} with {len(current_data)} records")
        print(f"Bounds: lat [{bounds['min_latitude']:.6f}, {bounds['max_latitude']:.6f}], lon [{bounds['min_longitude']:.6f}, {bounds['max_longitude']:.6f}]")
        
        # Check response size and prepare data
        try:
            # First ensure all numeric values are JSON-serializable
            response_df = current_data.copy()
            
            # Clean data again to be absolutely sure
            response_df = clean_data_for_json(response_df)
            
            # Convert to dict with extra safety
            try:
                response_data = response_df.to_dict(orient='records')
            except Exception as e:
                print(f"Initial to_dict failed: {e}")
                # If that fails, try a more aggressive approach
                response_data = []
                for idx, row in response_df.iterrows():
                    row_dict = {}
                    for col, val in row.items():
                        if pd.isna(val):
                            row_dict[col] = None
                        elif isinstance(val, (np.integer, np.floating)):
                            row_dict[col] = float(val)
                        elif isinstance(val, np.bool_):
                            row_dict[col] = bool(val)
                        else:
                            row_dict[col] = val
                    response_data.append(row_dict)
            
            # Verify JSON serialization will work
            try:
                import json
                _ = json.dumps(response_data)
            except Exception as e:
                print(f"JSON serialization test failed: {e}")
                # Find the problematic value
                for i, record in enumerate(response_data):
                    try:
                        json.dumps(record)
                    except Exception as record_error:
                        print(f"Problem in record {i}: {record_error}")
                        print(f"Record keys: {list(record.keys())}")
                        # Try to find the specific problematic field
                        for key, value in record.items():
                            try:
                                json.dumps({key: value})
                            except:
                                print(f"Problem field: {key} = {value} (type: {type(value)})")
                                # Fix it
                                if isinstance(value, float) and np.isnan(value):
                                    record[key] = None
                                elif isinstance(value, (np.integer, np.floating)):
                                    record[key] = None if np.isnan(value) else float(value)
                                else:
                                    record[key] = str(value)
            
            response_size = len(str(response_data))
            print(f"Response size: {response_size / 1024 / 1024:.2f} MB")
            
            if response_size > 50 * 1024 * 1024:  # 50MB limit
                print("WARNING: Response size is very large, might cause issues")
                # Sample the data if it's too large
                sample_rate = int(len(response_df) / 5000)  # Target ~5000 points
                if sample_rate > 1:
                    print(f"Sampling data: taking every {sample_rate}th point")
                    response_data = response_data[::sample_rate]
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error preparing response data: {str(e)}'})
        
        return jsonify({
            'success': True,
            'scenario': new_scenario,
            'data': response_data,
            'bounds': bounds
        })
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"ERROR in switch_scenario: {error_msg}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}. Check console for details.'})

@app.route('/api/performance_metrics')
def get_performance_metrics():
    """API endpoint for real-time performance metrics"""
    global current_data
    
    if current_data is None:
        return jsonify({"error": "No data loaded"})
    
    try:
        metrics = {}
        
        # ALVIRA metrics
        if 'alvira_pos_error_m' in current_data.columns:
            alvira_valid = current_data.dropna(subset=['alvira_pos_error_m'])
            if len(alvira_valid) > 0:
                metrics['alvira'] = {
                    'detections': len(alvira_valid),
                    'detection_rate': len(alvira_valid) / len(current_data) * 100,
                    'mean_pos_error': float(alvira_valid['alvira_pos_error_m'].mean()),
                    'max_pos_error': float(alvira_valid['alvira_pos_error_m'].max()),
                    'mean_alt_error': float(alvira_valid['alvira_alt_error_m'].mean()) if 'alvira_alt_error_m' in alvira_valid.columns else 0
                }
        
        # ARCUS metrics
        if 'arcus_pos_error_m' in current_data.columns:
            arcus_valid = current_data.dropna(subset=['arcus_pos_error_m'])
            if len(arcus_valid) > 0:
                metrics['arcus'] = {
                    'detections': len(arcus_valid),
                    'detection_rate': len(arcus_valid) / len(current_data) * 100,
                    'mean_pos_error': float(arcus_valid['arcus_pos_error_m'].mean()),
                    'max_pos_error': float(arcus_valid['arcus_pos_error_m'].max()),
                    'mean_alt_error': float(arcus_valid['arcus_alt_error_m'].mean()) if 'arcus_alt_error_m' in arcus_valid.columns else 0
                }
        
        # DIANA metrics
        if 'diana_snr' in current_data.columns:
            diana_valid = current_data.dropna(subset=['diana_snr'])
            if len(diana_valid) > 0:
                metrics['diana'] = {
                    'detections': len(diana_valid),
                    'detection_rate': len(diana_valid) / len(current_data) * 100,
                    'mean_snr': float(diana_valid['diana_snr'].mean()),
                    'max_range': float(diana_valid['diana_range'].max()) if 'diana_range' in diana_valid.columns else 0
                }
        
        # VENUS metrics
        if 'venus_frequency' in current_data.columns:
            venus_valid = current_data.dropna(subset=['venus_frequency'])
            if len(venus_valid) > 0:
                metrics['venus'] = {
                    'detections': len(venus_valid),
                    'detection_rate': len(venus_valid) / len(current_data) * 100,
                    'mean_frequency': float(venus_valid['venus_frequency'].mean())
                }
        
        return jsonify(metrics)
        
    except Exception as e:
        return jsonify({"error": f"Error calculating metrics: {e}"})

@app.route('/api/status')
def get_status():
    """System status endpoint"""
    global current_data, current_scenario
    
    if current_data is None:
        return jsonify({"status": "no_data", "message": "No data loaded"})
    
    return jsonify({
        "status": "active",
        "current_scenario": current_scenario,
        "total_records": len(current_data),
        "sensors_active": len([col for col in current_data.columns if any(sensor in col for sensor in ['alvira', 'arcus', 'diana', 'venus'])]),
        "time_range": f"{current_data['datetime(utc)'].min()} to {current_data['datetime(utc)'].max()}"
    })

@app.route('/api/debug_scenario/<scenario_name>')
def debug_scenario(scenario_name):
    """Debug endpoint to see what's causing JSON issues"""
    try:
        # Load the scenario data
        data = load_scenario_data(scenario_name)
        if data is None:
            return jsonify({'error': 'Failed to load scenario'})
        
        # Find problematic columns
        problems = {}
        
        for col in data.columns:
            col_data = data[col]
            
            # Check for various issues
            if col_data.dtype in ['float64', 'float32', 'int64', 'int32']:
                # Check for NaN
                nan_count = col_data.isna().sum()
                if nan_count > 0:
                    problems[col] = problems.get(col, {})
                    problems[col]['nan_count'] = int(nan_count)
                
                # Check for infinity
                if col_data.dtype in ['float64', 'float32']:
                    inf_count = np.isinf(col_data).sum()
                    if inf_count > 0:
                        problems[col] = problems.get(col, {})
                        problems[col]['inf_count'] = int(inf_count)
                
                # Check first few values
                problems[col] = problems.get(col, {})
                problems[col]['sample_values'] = [
                    str(v) for v in col_data.head(5).tolist()
                ]
        
        # Try to identify the specific "App_Tip" column
        app_tip_cols = [col for col in data.columns if 'App_Tip' in col]
        
        return jsonify({
            'scenario': scenario_name,
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'problematic_columns': problems,
            'app_tip_columns': app_tip_cols,
            'column_dtypes': {col: str(data[col].dtype) for col in data.columns}
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@app.route('/api/test_scenario/<scenario_name>')
def test_scenario(scenario_name):
    """Test endpoint to debug scenario loading"""
    print(f"\n=== Testing scenario: {scenario_name} ===")
    
    # Find the scenario path
    possible_paths = [
        f'data/src/icmcis-drone-detection/train/train/{scenario_name}',
        f'Data/src/icmcis-drone-detection/train/train/{scenario_name}'
    ]
    
    scenario_path = None
    for path in possible_paths:
        if os.path.exists(path):
            scenario_path = path
            break
    
    if not scenario_path:
        return jsonify({
            'error': 'Scenario path not found',
            'tried_paths': possible_paths
        })
    
    # List files in the scenario
    files = os.listdir(scenario_path)
    print(f"Files in {scenario_path}:")
    for f in files:
        print(f"  - {f}")
    
    # Try to load ground truth
    processor = MultiSensorDataProcessor(scenario_path)
    gt_files = processor.find_ground_truth_files()
    
    result = {
        'scenario': scenario_name,
        'path': scenario_path,
        'files': files,
        'ground_truth_files': [os.path.basename(f) for f in gt_files],
        'sensor_files': {}
    }
    
    # Check each sensor file
    for sensor in ['ALVIRA', 'ARCUS', 'DIANA', 'VENUS']:
        sensor_file = os.path.join(scenario_path, f'{sensor}_scenario.csv')
        if os.path.exists(sensor_file):
            try:
                df = pd.read_csv(sensor_file)
                result['sensor_files'][sensor] = {
                    'exists': True,
                    'rows': len(df),
                    'columns': list(df.columns)[:10]  # First 10 columns
                }
            except Exception as e:
                result['sensor_files'][sensor] = {
                    'exists': True,
                    'error': str(e)
                }
        else:
            result['sensor_files'][sensor] = {'exists': False}
    
    # Try to load the scenario
    try:
        data = load_scenario_data(scenario_name)
        if data is not None:
            result['load_status'] = 'success'
            result['total_records'] = len(data)
            result['columns'] = list(data.columns)
            result['has_sensor_data'] = any(sensor in ' '.join(data.columns) for sensor in ['alvira', 'arcus', 'diana', 'venus'])
        else:
            result['load_status'] = 'failed'
    except Exception as e:
        result['load_status'] = 'error'
        result['load_error'] = str(e)
    
    return jsonify(result)

if __name__ == '__main__':
    print("🚀 Starting Multi-Scenario Dashboard")
    print("📁 Looking for scenarios in 'data/src/icmcis-drone-detection/train/train'")
    app.run(debug=True, host='0.0.0.0', port=5000)