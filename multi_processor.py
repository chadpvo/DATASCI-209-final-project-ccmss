import pandas as pd
import numpy as np
import json
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime
from flask import Flask, render_template, jsonify
from IPython.display import display, HTML, IFrame
import warnings
warnings.filterwarnings('ignore')

class MultiSensorDataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ground_truth = None
        self.sensor_data = {}
        self.merged_data = None

    def load_all_data(self):
        """Load all sensor data files"""
        print("Loading multi-sensor drone detection data...")

        # Define expected files
        files_to_load = {
            'ground_truth': "drone_data/2020-09-29_14-10-56_v2.csv",
            'ALVIRA': 'drone_data/ALVIRA_scenario.csv',
            'ARCUS': 'drone_data/ARCUS_scenario.csv',
            'DIANA': 'drone_data/DIANA_scenario.csv',
            'VENUS': 'drone_data/VENUS_scenario.csv'
        }

        # Load ground truth (drone log)
        gt_file = os.path.join(self.data_dir, files_to_load['ground_truth'])
        if os.path.exists(gt_file):
            self.ground_truth = pd.read_csv(gt_file)
            self.ground_truth['datetime(utc)'] = pd.to_datetime(self.ground_truth['datetime(utc)'])
            print(f"Ground truth loaded: {len(self.ground_truth):,} records")
        else:
            print(f"Ground truth file not found: {gt_file}")
            return False

        # Load sensor data
        sensors_loaded = 0
        for sensor_name, filename in files_to_load.items():
            if sensor_name == 'ground_truth':
                continue

            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    df['datetime(utc)'] = pd.to_datetime(df['datetime(utc)'])
                    self.sensor_data[sensor_name] = df
                    sensors_loaded += 1
                    print(f"{sensor_name} loaded: {len(df):,} records")
                except Exception as e:
                    print(f"Error loading {sensor_name}: {e}")
            else:
                print(f"{sensor_name} file not found: {filepath}")

        if sensors_loaded > 0:
            print(f"\nSuccessfully loaded {sensors_loaded} sensor datasets!")
            return True
        else:
            print("No sensor data files found")
            return False

    def merge_sensor_data(self):
        """Merge all sensor data with ground truth using time alignment"""
        if self.ground_truth is None:
            print("No ground truth data available")
            return None

        print("Merging sensor data with ground truth...")

        merged = self.ground_truth.copy()

        column_mapping = {
            'latitude': 'gt_latitude',
            'longitude': 'gt_longitude',
            'altitude(m)': 'gt_altitude',
            'velocityX(mps)': 'gt_vel_x',
            'velocityY(mps)': 'gt_vel_y',
            'velocityZ(mps)': 'gt_vel_z',
            'speed(mps)': 'gt_speed'
        }

        existing_columns = {k: v for k, v in column_mapping.items() if k in merged.columns}
        merged = merged.rename(columns=existing_columns)
        print(f"Ground truth columns renamed: {list(existing_columns.values())}")

        if 'ALVIRA' in self.sensor_data:
            alvira = self.sensor_data['ALVIRA']
            alvira_clean = alvira[alvira['AlviraTracksTrackPosition_Latitude'].notna()].copy()

            if len(alvira_clean) > 0:
                alvira_clean = alvira_clean.rename(columns={
                    'AlviraTracksTrackPosition_Latitude': 'alvira_latitude',
                    'AlviraTracksTrackPosition_Longitude': 'alvira_longitude',
                    'AlviraTracksTrackPosition_Altitude': 'alvira_altitude',
                    'AlviraTracksTrackVelocity_Speed': 'alvira_speed',
                    'AlviraTracksTrack_Classification': 'alvira_classification',
                    'AlviraTracksTrack_Score': 'alvira_score'
                })

                merged = pd.merge_asof(
                    merged.sort_values('datetime(utc)'),
                    alvira_clean.sort_values('datetime(utc)')[['datetime(utc)', 'alvira_latitude', 'alvira_longitude', 'alvira_altitude', 'alvira_speed', 'alvira_classification', 'alvira_score']],
                    on='datetime(utc)',
                    direction='nearest',
                    tolerance=pd.Timedelta('5s')
                )
                print("ALVIRA data merged")
            else:
                print("No valid ALVIRA tracking data found")

        if 'ARCUS' in self.sensor_data:
            arcus = self.sensor_data['ARCUS']
            arcus_clean = arcus[arcus['ArcusTracksTrackPosition_Latitude'].notna()].copy()

            if len(arcus_clean) > 0:
                arcus_clean = arcus_clean.rename(columns={
                    'ArcusTracksTrackPosition_Latitude': 'arcus_latitude',
                    'ArcusTracksTrackPosition_Longitude': 'arcus_longitude',
                    'ArcusTracksTrackPosition_Altitude': 'arcus_altitude',
                    'ArcusTracksTrackVelocity_Speed': 'arcus_speed',
                    'ArcusTracksTrack_Classification': 'arcus_classification',
                    'ArcusTracksTrack_Score': 'arcus_score'
                })

                merged = pd.merge_asof(
                    merged.sort_values('datetime(utc)'),
                    arcus_clean.sort_values('datetime(utc)')[['datetime(utc)', 'arcus_latitude', 'arcus_longitude', 'arcus_altitude', 'arcus_speed', 'arcus_classification', 'arcus_score']],
                    on='datetime(utc)',
                    direction='nearest',
                    tolerance=pd.Timedelta('5s')
                )
                print("ARCUS data merged")
            else:
                print("No valid ARCUS tracking data found")

        if 'DIANA' in self.sensor_data:
            diana = self.sensor_data['DIANA']
            diana_clean = diana[diana['DianaTargetsTargetSignal_bearing_deg'].notna()].copy()

            if len(diana_clean) > 0:
                diana_clean = diana_clean.rename(columns={
                    'DianaTargetsTargetSignal_bearing_deg': 'diana_bearing',
                    'DianaTargetsTargetSignal_range_m': 'diana_range',
                    'DianaTargetsTargetSignal_snr_dB': 'diana_snr',
                    'DianaTargetsTargetClassification_type': 'diana_classification',
                    'DianaTargetsTargetClassification_score': 'diana_score'
                })

                merged = pd.merge_asof(
                    merged.sort_values('datetime(utc)'),
                    diana_clean.sort_values('datetime(utc)')[['datetime(utc)', 'diana_bearing', 'diana_range', 'diana_snr', 'diana_classification', 'diana_score']],
                    on='datetime(utc)',
                    direction='nearest',
                    tolerance=pd.Timedelta('10s')
                )
                print("DIANA data merged")
            else:
                print("No valid DIANA signal data found")

        if 'VENUS' in self.sensor_data:
            venus = self.sensor_data['VENUS']
            venus_clean = venus[venus['VenusTrigger_Azimuth'].notna()].copy()

            if len(venus_clean) > 0:
                venus_clean = venus_clean.rename(columns={
                    'VenusTrigger_Azimuth': 'venus_azimuth',
                    'VenusTrigger_Frequency': 'venus_frequency',
                    'VenusTriggerVenusName_isThreat': 'venus_threat_score'
                })

                merged = pd.merge_asof(
                    merged.sort_values('datetime(utc)'),
                    venus_clean.sort_values('datetime(utc)')[['datetime(utc)', 'venus_azimuth', 'venus_frequency', 'venus_threat_score']],
                    on='datetime(utc)',
                    direction='nearest',
                    tolerance=pd.Timedelta('10s')
                )
                print("VENUS data merged")
            else:
                print("No valid VENUS signal data found")

        print("Calculating position errors...")
        if 'alvira_latitude' in merged.columns:
            merged['alvira_pos_error_m'] = np.sqrt(
                ((merged['gt_latitude'] - merged['alvira_latitude']) * 111000) ** 2 +
                ((merged['gt_longitude'] - merged['alvira_longitude']) * 111000) ** 2
            )
            merged['alvira_alt_error_m'] = abs(merged['gt_altitude'] - merged['alvira_altitude'])
            print("ALVIRA position errors calculated")

        if 'arcus_latitude' in merged.columns:
            merged['arcus_pos_error_m'] = np.sqrt(
                ((merged['gt_latitude'] - merged['arcus_latitude']) * 111000) ** 2 +
                ((merged['gt_longitude'] - merged['arcus_longitude']) * 111000) ** 2
            )
            merged['arcus_alt_error_m'] = abs(merged['gt_altitude'] - merged['arcus_altitude'])
            print("ARCUS position errors calculated")

        merged = merged.sort_values('datetime(utc)').reset_index(drop=True)
        self.merged_data = merged
        print(f"\nSensor data fusion complete!")
        print(f"Final dataset: {len(merged):,} synchronized records")
        self.print_performance_summary()
        return merged

    def print_performance_summary(self):
        if self.merged_data is None:
            return

        print("\n" + "="*60)
        print("SENSOR PERFORMANCE SUMMARY")
        print("="*60)

        total_records = len(self.merged_data)
        time_span = (self.merged_data['datetime(utc)'].max() - self.merged_data['datetime(utc)'].min()).total_seconds() / 60
        print(f"Total Records: {total_records:,}")
        print(f"Time Span: {time_span:.1f} minutes")

        if 'alvira_pos_error_m' in self.merged_data.columns:
            alvira_valid = self.merged_data.dropna(subset=['alvira_pos_error_m'])
            if len(alvira_valid) > 0:
                print(f"\nALVIRA (2D Radar):")
                print(f"   Detection Rate:      {len(alvira_valid)/total_records*100:.1f}% ({len(alvira_valid):,} detections)")
                print(f"   Avg Position Error:  {alvira_valid['alvira_pos_error_m'].mean():.1f} ± {alvira_valid['alvira_pos_error_m'].std():.1f} m")
                print(f"   Max Position Error:  {alvira_valid['alvira_pos_error_m'].max():.1f} m")
                print(f"   Avg Altitude Error:  {alvira_valid['alvira_alt_error_m'].mean():.1f} m")

        if 'arcus_pos_error_m' in self.merged_data.columns:
            arcus_valid = self.merged_data.dropna(subset=['arcus_pos_error_m'])
            if len(arcus_valid) > 0:
                print(f"\nARCUS (3D Radar):")
                print(f"   Detection Rate:      {len(arcus_valid)/total_records*100:.1f}% ({len(arcus_valid):,} detections)")
                print(f"   Avg Position Error:  {arcus_valid['arcus_pos_error_m'].mean():.1f} ± {arcus_valid['arcus_pos_error_m'].std():.1f} m")
                print(f"   Max Position Error:  {arcus_valid['arcus_pos_error_m'].max():.1f} m")
                print(f"   Avg Altitude Error:  {arcus_valid['arcus_alt_error_m'].mean():.1f} m")

        if 'diana_snr' in self.merged_data.columns:
            diana_valid = self.merged_data.dropna(subset=['diana_snr'])
            if len(diana_valid) > 0:
                print(f"\nDIANA (RF Direction Finding):")
                print(f"   Detection Rate:      {len(diana_valid)/total_records*100:.1f}% ({len(diana_valid):,} detections)")
                print(f"   Avg SNR:            {diana_valid['diana_snr'].mean():.1f} dB")
                if 'diana_range' in diana_valid.columns:
                    print(f"   Max Range:          {diana_valid['diana_range'].max():.0f} m")

        if 'venus_frequency' in self.merged_data.columns:
            venus_valid = self.merged_data.dropna(subset=['venus_frequency'])
            if len(venus_valid) > 0:
                print(f"\nVENUS (RF Direction Finding):")
                print(f"   Detection Rate:      {len(venus_valid)/total_records*100:.1f}% ({len(venus_valid):,} detections)")
                print(f"   Avg Frequency:      {venus_valid['venus_frequency'].mean()/1e6:.0f} MHz")

        print("="*60)
