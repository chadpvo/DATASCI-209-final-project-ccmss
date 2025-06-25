def create_flask_app(merged_data):
    """Create Flask app with the processed data"""
    
    from flask import Flask, render_template, jsonify
    import json
    
    app = Flask(__name__)
    
    # Prepare data for frontend (convert datetime to string)
    display_data = merged_data.copy()
    display_data['datetime(utc)'] = display_data['datetime(utc)'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # Calculate visualization bounds
    buffer = 0.001  # Add small buffer for better visualization
    bounds = {
        'min_latitude': float(display_data['gt_latitude'].min()) - buffer,
        'max_latitude': float(display_data['gt_latitude'].max()) + buffer, 
        'min_longitude': float(display_data['gt_longitude'].min()) - buffer,
        'max_longitude': float(display_data['gt_longitude'].max()) + buffer,
        'min_altitude': float(display_data['gt_altitude'].min()),
        'max_altitude': float(display_data['gt_altitude'].max()),
        'min_time': display_data['datetime(utc)'].min(),
        'max_time': display_data['datetime(utc)'].max(),
        'total_frames': len(display_data)
    }
    
    print(f" Dashboard bounds calculated:")
    print(f"   Latitude: {bounds['min_latitude']:.6f} to {bounds['max_latitude']:.6f}")
    print(f"   Longitude: {bounds['min_longitude']:.6f} to {bounds['max_longitude']:.6f}")
    print(f"   Altitude: {bounds['min_altitude']:.1f} to {bounds['max_altitude']:.1f} m")
    print(f"   Time span: {bounds['min_time']} to {bounds['max_time']}")
    
    @app.route('/')
    def dashboard():
        """Main dashboard route"""
        try:
            # Convert data to JSON for frontend
            dashboard_data = display_data.to_dict(orient='records')
            
            return render_template(
                'fusion_dashboard.html',
                drone_data_json=json.dumps(dashboard_data),
                **bounds
            )
        except Exception as e:
            return f"<h1>Dashboard Error</h1><p>Error loading dashboard: {e}</p>"
    
    @app.route('/api/performance_metrics')
    def get_performance_metrics():
        """API endpoint for real-time performance metrics"""
        try:
            metrics = {}
            
            # ALVIRA metrics
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
            
            # ARCUS metrics
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
            
            # DIANA metrics
            if 'diana_snr' in display_data.columns:
                diana_valid = display_data.dropna(subset=['diana_snr'])
                if len(diana_valid) > 0:
                    metrics['diana'] = {
                        'detections': len(diana_valid),
                        'detection_rate': len(diana_valid) / len(display_data) * 100,
                        'mean_snr': float(diana_valid['diana_snr'].mean()),
                        'max_range': float(diana_valid['diana_range'].max()) if 'diana_range' in diana_valid.columns else 0
                    }
            
            # VENUS metrics
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
            return jsonify({"error": f"Error calculating metrics: {e}"})
    
    @app.route('/api/status')
    def get_status():
        """System status endpoint"""
        return jsonify({
            "status": "active",
            "total_records": len(display_data),
            "sensors_active": len([col for col in display_data.columns if any(sensor in col for sensor in ['alvira', 'arcus', 'diana', 'venus'])]),
            "time_range": f"{bounds['min_time']} to {bounds['max_time']}"
        })
    
    return app

import pandas as pd

merged_data = pd.read_csv("data/merged_data.csv", parse_dates=['datetime(utc)'])
app = create_flask_app(merged_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)