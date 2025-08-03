import os
import pandas as pd
import numpy as np
import glob

def clean_data_for_json(df):
    """Clean dataframe to ensure JSON serialization compatibility"""
    df_clean = df.copy()
    
    # First, let's check all columns
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # For object columns, convert any numpy types to Python types
            df_clean[col] = df_clean[col].apply(lambda x: 
                None if pd.isna(x) else 
                str(x) if isinstance(x, (np.integer, np.floating, np.ndarray)) else 
                x
            )
        elif np.issubdtype(df_clean[col].dtype, np.number):
            # For numeric columns
            # First replace infinity with NaN
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            
            # Then handle NaN values
            if df_clean[col].isna().any():
                # Convert to nullable float type first
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except:
                    pass
                
                # Replace NaN with None
                df_clean[col] = df_clean[col].where(pd.notnull(df_clean[col]), None)
    
    # Final check - convert any remaining numpy types
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Ensure it's a proper pandas series with no numpy scalar types
            df_clean[col] = df_clean[col].astype(object).where(pd.notnull(df_clean[col]), None)
    
    return df_clean

class MultiSensorDataProcessor:
    def __init__(self, scenario_path):
        self.scenario_path = scenario_path
        self.ground_truth = None
        self.sensor_data = {}
        self.merged_data = None
    
    def find_ground_truth_files(self):
        """Find ground truth files with various naming patterns"""
        # Try different patterns for ground truth files
        patterns = [
            "*_v2.csv",
            "*corrected.csv",
            "*Airdata*.csv"
        ]
        
        gt_files = []
        for pattern in patterns:
            files = glob.glob(os.path.join(self.scenario_path, pattern))
            gt_files.extend(files)
        
        # Filter out sensor files that might match these patterns
        gt_files = [f for f in gt_files if not any(sensor in f for sensor in ['ALVIRA', 'ARCUS', 'DIANA', 'VENUS'])]
        
        return sorted(list(set(gt_files)))  # Remove duplicates and sort
    
    def load_all_data(self):
        """Load all sensor data files for the scenario"""
        print(f"Loading data from: {self.scenario_path}")
        
        # Find ground truth files
        gt_files = self.find_ground_truth_files()
        
        if not gt_files:
            print("No ground truth file found")
            return False
        
        print(f"Found {len(gt_files)} ground truth file(s)")
        
        # For scenarios with multiple drones, combine the ground truth data
        if len(gt_files) > 1:
            print("Multiple ground truth files found - combining them")
            gt_dfs = []
            for gt_file in gt_files:
                try:
                    df = pd.read_csv(gt_file, low_memory=False)
                    df['datetime(utc)'] = pd.to_datetime(df['datetime(utc)'])
                    gt_dfs.append(df)
                    print(f"  - Loaded {os.path.basename(gt_file)}: {len(df)} records")
                except Exception as e:
                    print(f"  - Error loading {os.path.basename(gt_file)}: {e}")
            
            if gt_dfs:
                # Combine all ground truth data and sort by time
                self.ground_truth = pd.concat(gt_dfs, ignore_index=True)
                self.ground_truth = self.ground_truth.sort_values('datetime(utc)').reset_index(drop=True)
                print(f"Combined ground truth: {len(self.ground_truth)} total records")
            else:
                print("Failed to load any ground truth files")
                return False
        else:
            # Single ground truth file
            try:
                # Load with low_memory=False to avoid dtype warnings
                self.ground_truth = pd.read_csv(gt_files[0], low_memory=False)
                self.ground_truth['datetime(utc)'] = pd.to_datetime(self.ground_truth['datetime(utc)'])
                print(f"Ground truth loaded from {os.path.basename(gt_files[0])}: {len(self.ground_truth)} records")
            except Exception as e:
                print(f"Error loading ground truth: {e}")
                return False
        
        # Check if required columns exist
        required_cols = ['datetime(utc)', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in self.ground_truth.columns]
        if missing_cols:
            print(f"Missing required columns in ground truth: {missing_cols}")
            print(f"Available columns: {list(self.ground_truth.columns)}")
            
            # Try to find alternative column names
            col_mapping = {
                'latitude': ['lat', 'Latitude', 'GPS_latitude', 'gps_lat'],
                'longitude': ['lon', 'lng', 'Longitude', 'GPS_longitude', 'gps_lon', 'gps_lng'],
                'datetime(utc)': ['datetime', 'timestamp', 'time', 'UTC_time']
            }
            
            for required, alternatives in col_mapping.items():
                if required not in self.ground_truth.columns:
                    for alt in alternatives:
                        if alt in self.ground_truth.columns:
                            print(f"Found alternative column '{alt}' for '{required}'")
                            self.ground_truth[required] = self.ground_truth[alt]
                            break
            
            # Check again
            missing_cols = [col for col in required_cols if col not in self.ground_truth.columns]
            if missing_cols:
                print(f"Still missing columns after trying alternatives: {missing_cols}")
                return False
        
        # Load sensor files
        sensor_files = {
            'ALVIRA': 'ALVIRA_scenario.csv',
            'ARCUS': 'ARCUS_scenario.csv',
            'DIANA': 'DIANA_scenario.csv',
            'VENUS': 'VENUS_scenario.csv'
        }
        
        sensors_loaded = 0
        for sensor_name, filename in sensor_files.items():
            filepath = os.path.join(self.scenario_path, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    if 'datetime(utc)' in df.columns:
                        df['datetime(utc)'] = pd.to_datetime(df['datetime(utc)'])
                        self.sensor_data[sensor_name] = df
                        sensors_loaded += 1
                        print(f"{sensor_name} loaded: {len(df)} records")
                    else:
                        print(f"{sensor_name} file missing datetime(utc) column")
                except Exception as e:
                    print(f"Error loading {sensor_name}: {e}")
            else:
                print(f"{sensor_name} file not found: {filepath}")
        
        if sensors_loaded == 0:
            print("Warning: No sensor data files loaded, using ground truth only")
        
        return True
    
    def merge_sensor_data(self):
        """Merge all sensor data with ground truth using time alignment"""
        if self.ground_truth is None:
            return None
        
        print("Merging sensor data with ground truth...")
        
        # Start with ground truth
        merged = self.ground_truth.copy()
        
        # Ensure altitude column exists
        if 'altitude(m)' not in merged.columns and 'altitude' in merged.columns:
            merged['altitude(m)'] = merged['altitude']
        elif 'altitude(m)' not in merged.columns:
            print("Warning: No altitude column found in ground truth")
            merged['altitude(m)'] = 0  # Default altitude
        
        # Rename ground truth columns
        column_mapping = {
            'latitude': 'gt_latitude',
            'longitude': 'gt_longitude',
            'altitude(m)': 'gt_altitude',
            'altitude': 'gt_altitude',  # Alternative altitude column name
            'velocityX(mps)': 'gt_vel_x',
            'velocityY(mps)': 'gt_vel_y',
            'velocityZ(mps)': 'gt_vel_z',
            'speed(mps)': 'gt_speed'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in merged.columns and new_col not in merged.columns:
                merged[new_col] = merged[old_col]
        
        # Ensure we have the required ground truth columns
        if 'gt_latitude' not in merged.columns or 'gt_longitude' not in merged.columns:
            print("Error: Missing required latitude/longitude columns after renaming")
            print(f"Available columns: {list(merged.columns)}")
            return None
        
        # If gt_altitude is missing, create it with default value
        if 'gt_altitude' not in merged.columns:
            print("Warning: gt_altitude column not found, using default value of 0")
            merged['gt_altitude'] = 0
        
        # Sort by time before merging
        merged = merged.sort_values('datetime(utc)').reset_index(drop=True)
        
        # Merge ALVIRA data
        if 'ALVIRA' in self.sensor_data:
            alvira = self.sensor_data['ALVIRA']
            
            # Check which ALVIRA columns are available
            alvira_lat_col = None
            alvira_lon_col = None
            for col in alvira.columns:
                if 'Latitude' in col and alvira_lat_col is None:
                    alvira_lat_col = col
                if 'Longitude' in col and alvira_lon_col is None:
                    alvira_lon_col = col
            
            if alvira_lat_col and alvira_lon_col:
                alvira_clean = alvira[alvira[alvira_lat_col].notna()].copy()
                
                if len(alvira_clean) > 0:
                    # Find all relevant ALVIRA columns
                    rename_dict = {
                        alvira_lat_col: 'alvira_latitude',
                        alvira_lon_col: 'alvira_longitude'
                    }
                    
                    # Look for altitude, speed, classification, score columns
                    for col in alvira_clean.columns:
                        if 'Altitude' in col and 'alvira_altitude' not in rename_dict.values():
                            rename_dict[col] = 'alvira_altitude'
                        elif 'Speed' in col and 'alvira_speed' not in rename_dict.values():
                            rename_dict[col] = 'alvira_speed'
                        elif 'Classification' in col and 'alvira_classification' not in rename_dict.values():
                            rename_dict[col] = 'alvira_classification'
                        elif 'Score' in col and 'alvira_score' not in rename_dict.values():
                            rename_dict[col] = 'alvira_score'
                    
                    alvira_clean = alvira_clean.rename(columns=rename_dict)
                    
                    # Select columns that exist
                    merge_cols = ['datetime(utc)'] + [col for col in rename_dict.values() if col in alvira_clean.columns]
                    
                    merged = pd.merge_asof(
                        merged.sort_values('datetime(utc)'),
                        alvira_clean.sort_values('datetime(utc)')[merge_cols],
                        on='datetime(utc)',
                        direction='nearest',
                        tolerance=pd.Timedelta('5s')
                    )
                    print(f"ALVIRA data merged: {len(alvira_clean[alvira_clean['alvira_latitude'].notna()])} valid detections")
                else:
                    print("No valid ALVIRA tracking data found")
            else:
                print("ALVIRA data missing required position columns")
        
        # Merge ARCUS data (similar pattern)
        if 'ARCUS' in self.sensor_data:
            arcus = self.sensor_data['ARCUS']
            
            # Check which ARCUS columns are available
            arcus_lat_col = None
            arcus_lon_col = None
            for col in arcus.columns:
                if 'Latitude' in col and arcus_lat_col is None:
                    arcus_lat_col = col
                if 'Longitude' in col and arcus_lon_col is None:
                    arcus_lon_col = col
            
            if arcus_lat_col and arcus_lon_col:
                arcus_clean = arcus[arcus[arcus_lat_col].notna()].copy()
                
                if len(arcus_clean) > 0:
                    rename_dict = {
                        arcus_lat_col: 'arcus_latitude',
                        arcus_lon_col: 'arcus_longitude'
                    }
                    
                    for col in arcus_clean.columns:
                        if 'Altitude' in col and 'arcus_altitude' not in rename_dict.values():
                            rename_dict[col] = 'arcus_altitude'
                        elif 'Speed' in col and 'arcus_speed' not in rename_dict.values():
                            rename_dict[col] = 'arcus_speed'
                        elif 'Classification' in col and 'arcus_classification' not in rename_dict.values():
                            rename_dict[col] = 'arcus_classification'
                        elif 'Score' in col and 'arcus_score' not in rename_dict.values():
                            rename_dict[col] = 'arcus_score'
                    
                    arcus_clean = arcus_clean.rename(columns=rename_dict)
                    
                    merge_cols = ['datetime(utc)'] + [col for col in rename_dict.values() if col in arcus_clean.columns]
                    
                    merged = pd.merge_asof(
                        merged.sort_values('datetime(utc)'),
                        arcus_clean.sort_values('datetime(utc)')[merge_cols],
                        on='datetime(utc)',
                        direction='nearest',
                        tolerance=pd.Timedelta('5s')
                    )
                    print(f"ARCUS data merged: {len(arcus_clean[arcus_clean['arcus_latitude'].notna()])} valid detections")
                else:
                    print("No valid ARCUS tracking data found")
            else:
                print("ARCUS data missing required position columns")
        
        # Merge DIANA data
        if 'DIANA' in self.sensor_data:
            diana = self.sensor_data['DIANA']
            
            # Find bearing column
            diana_bearing_col = None
            for col in diana.columns:
                if 'bearing' in col.lower() or 'azimuth' in col.lower():
                    diana_bearing_col = col
                    break
            
            if diana_bearing_col:
                diana_clean = diana[diana[diana_bearing_col].notna()].copy()
                
                if len(diana_clean) > 0:
                    rename_dict = {diana_bearing_col: 'diana_bearing'}
                    
                    for col in diana_clean.columns:
                        if 'range' in col.lower() and 'diana_range' not in rename_dict.values():
                            rename_dict[col] = 'diana_range'
                        elif 'snr' in col.lower() and 'diana_snr' not in rename_dict.values():
                            rename_dict[col] = 'diana_snr'
                        elif 'classification' in col.lower() and 'diana_classification' not in rename_dict.values():
                            rename_dict[col] = 'diana_classification'
                        elif 'score' in col.lower() and 'diana_score' not in rename_dict.values():
                            rename_dict[col] = 'diana_score'
                    
                    diana_clean = diana_clean.rename(columns=rename_dict)
                    
                    merge_cols = ['datetime(utc)'] + [col for col in rename_dict.values() if col in diana_clean.columns]
                    
                    merged = pd.merge_asof(
                        merged.sort_values('datetime(utc)'),
                        diana_clean.sort_values('datetime(utc)')[merge_cols],
                        on='datetime(utc)',
                        direction='nearest',
                        tolerance=pd.Timedelta('10s')
                    )
                    print(f"DIANA data merged: {len(diana_clean[diana_clean['diana_bearing'].notna()])} valid detections")
        
        # Merge VENUS data
        if 'VENUS' in self.sensor_data:
            venus = self.sensor_data['VENUS']
            
            # Find azimuth/frequency columns
            venus_azimuth_col = None
            venus_freq_col = None
            for col in venus.columns:
                if 'azimuth' in col.lower() and venus_azimuth_col is None:
                    venus_azimuth_col = col
                if 'frequency' in col.lower() and venus_freq_col is None:
                    venus_freq_col = col
            
            if venus_azimuth_col or venus_freq_col:
                # Use whichever column exists for filtering
                filter_col = venus_azimuth_col if venus_azimuth_col else venus_freq_col
                venus_clean = venus[venus[filter_col].notna()].copy()
                
                if len(venus_clean) > 0:
                    rename_dict = {}
                    if venus_azimuth_col:
                        rename_dict[venus_azimuth_col] = 'venus_azimuth'
                    if venus_freq_col:
                        rename_dict[venus_freq_col] = 'venus_frequency'
                    
                    for col in venus_clean.columns:
                        if 'threat' in col.lower() and 'venus_threat_score' not in rename_dict.values():
                            rename_dict[col] = 'venus_threat_score'
                    
                    venus_clean = venus_clean.rename(columns=rename_dict)
                    
                    merge_cols = ['datetime(utc)'] + [col for col in rename_dict.values() if col in venus_clean.columns]
                    
                    merged = pd.merge_asof(
                        merged.sort_values('datetime(utc)'),
                        venus_clean.sort_values('datetime(utc)')[merge_cols],
                        on='datetime(utc)',
                        direction='nearest',
                        tolerance=pd.Timedelta('10s')
                    )
                    print(f"VENUS data merged: {len(venus_clean)} valid detections")
        
        # Calculate position errors only if sensor data exists
        if 'alvira_latitude' in merged.columns and 'alvira_longitude' in merged.columns:
            merged['alvira_pos_error_m'] = np.sqrt(
                ((merged['gt_latitude'] - merged['alvira_latitude']) * 111000) ** 2 +
                ((merged['gt_longitude'] - merged['alvira_longitude']) * 111000) ** 2
            )
            if 'alvira_altitude' in merged.columns and 'gt_altitude' in merged.columns:
                merged['alvira_alt_error_m'] = abs(merged['gt_altitude'] - merged['alvira_altitude'])
        
        if 'arcus_latitude' in merged.columns and 'arcus_longitude' in merged.columns:
            merged['arcus_pos_error_m'] = np.sqrt(
                ((merged['gt_latitude'] - merged['arcus_latitude']) * 111000) ** 2 +
                ((merged['gt_longitude'] - merged['arcus_longitude']) * 111000) ** 2
            )
            if 'arcus_altitude' in merged.columns and 'gt_altitude' in merged.columns:
                merged['arcus_alt_error_m'] = abs(merged['gt_altitude'] - merged['arcus_altitude'])
        
        # Sort by time
        merged = merged.sort_values('datetime(utc)').reset_index(drop=True)
        self.merged_data = merged
        
        print(f"Data fusion complete: {len(merged)} records")
        
        # Print summary of available data
        sensor_cols = [col for col in merged.columns if any(sensor in col for sensor in ['alvira', 'arcus', 'diana', 'venus'])]
        if sensor_cols:
            print(f"Available sensor columns: {len(sensor_cols)}")
            for sensor in ['alvira', 'arcus', 'diana', 'venus']:
                sensor_specific = [col for col in sensor_cols if sensor in col]
                if sensor_specific:
                    print(f"  - {sensor.upper()}: {', '.join(sensor_specific)}")
        else:
            print("Warning: No sensor data columns found in merged data")
        
        return merged

def load_scenario_data(scenario_name):
    """Load and process scenario data using the MultiSensorDataProcessor"""
    try:
        # Handle both drone_Data and drone_data directory names
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
            print(f"Scenario path not found for: {scenario_name}")
            print(f"Tried paths: {possible_paths}")
            return None
        
        # Use the processor to load and merge data
        processor = MultiSensorDataProcessor(scenario_path)
        
        if not processor.load_all_data():
            print(f"Failed to load data for scenario: {scenario_name}")
            return None
        
        merged_data = processor.merge_sensor_data()
        
        if merged_data is None:
            print(f"Failed to merge data for scenario: {scenario_name}")
            return None
        
        # Convert datetime to string for JSON serialization
        merged_data['datetime(utc)'] = merged_data['datetime(utc)'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        
        # Remove rows with NaN coordinates
        merged_data = merged_data.dropna(subset=['gt_latitude', 'gt_longitude'])
        
        # Clean data for JSON serialization
        merged_data = clean_data_for_json(merged_data)
        
        print(f"Processed {scenario_name}: {len(merged_data)} data points")
        return merged_data
        
    except Exception as e:
        print(f"Error loading scenario {scenario_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None