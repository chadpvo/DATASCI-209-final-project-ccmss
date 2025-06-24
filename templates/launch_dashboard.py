import requests
from multi_processor import MultiSensorDataProcessor, DATA_DIR
from flask_app import create_flask_app

def send_data_to_dashboard(merged_data):
    payload = merged_data.to_dict(orient='records')
    url = "https://drone-dashboard-yyrbx.ondigitalocean.app/"
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception:
        pass

def launch_dashboard():
    """Launch the complete multi-sensor fusion dashboard"""
    
    print("LAUNCHING MULTI-SENSOR FUSION DASHBOARD")
    print("="*55)
    
    # Step 1: Process the data
    print("Step 1: Processing multi-sensor data...")
    
    processor = MultiSensorDataProcessor(DATA_DIR)
    
    # Load data
    if not processor.load_all_data():
        print("Failed to load data. Please check your DATA_DIR path.")
        return None
    
    # Merge data
    merged_data = processor.merge_sensor_data()
    if merged_data is None:
        print("Failed to merge sensor data.")
        return None
    
    print(f"Data processing complete: {len(merged_data):,} records ready")
    
    # Step 2: Create Flask application
    print("\nStep 2: Creating Flask application...")
    
    app = create_flask_app(merged_data)
    
    # Step 3: Launch the dashboard
    print("\nStep 3: Starting dashboard server...")
    print(f"Dashboard URL: http://localhost:5000")
    print(f"Alternative: http://127.0.0.1:5000")
    
    print(f"\nDASHBOARD FEATURES:")
    print(f"   • Interactive timeline with {len(merged_data):,} data points")
    print(f"   • Multi-sensor altitude comparison")
    print(f"   • Position error analysis")
    print(f"   • Flight path visualization")
    print(f"   • Real-time performance metrics")
    print(f"   • Sensor status monitoring")
    
    print(f"\nCONTROLS:")
    print(f"   • Use timeline slider to navigate")
    print(f"   • Ctrl+C here: Stop the server")
    
    print(f"\nSENSOR SUMMARY:")
    sensor_count = 0
    if 'alvira_latitude' in merged_data.columns:
        alvira_detections = merged_data['alvira_latitude'].notna().sum()
        print(f"   ALVIRA: {alvira_detections:,} detections ({alvira_detections/len(merged_data)*100:.1f}%)")
        sensor_count += 1
    
    if 'arcus_latitude' in merged_data.columns:
        arcus_detections = merged_data['arcus_latitude'].notna().sum()
        print(f"   ARCUS: {arcus_detections:,} detections ({arcus_detections/len(merged_data)*100:.1f}%)")
        sensor_count += 1
    
    if 'diana_snr' in merged_data.columns:
        diana_detections = merged_data['diana_snr'].notna().sum()
        print(f"   DIANA: {diana_detections:,} detections ({diana_detections/len(merged_data)*100:.1f}%)")
        sensor_count += 1
    
    if 'venus_frequency' in merged_data.columns:
        venus_detections = merged_data['venus_frequency'].notna().sum()
        print(f"   VENUS: {venus_detections:,} detections ({venus_detections/len(merged_data)*100:.1f}%)")
        sensor_count += 1
    
    print(f"\n{sensor_count} sensors active and ready!")
    print(f"\n" + "="*55)
    print(f"READY! Open your browser to: http://localhost:5000")
    print(f"="*55)
    
    # Launch Flask app
    '''
    try:
        app.run(host='localhost', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print(f"\n\nDashboard stopped by user")
    except Exception as e:
        print(f"\nError starting dashboard: {e}")
        print(f"Try:")
        print(f"   • Check if port 5000 is already in use")
        print(f"   • Restart your Jupyter kernel")
        print(f"   • Run the cells again from the beginning")
        '''