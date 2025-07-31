import pandas as pd
import numpy as np

def calculate_positional_error(df, prefix):
    df[f'{prefix}_pos_error_m'] = np.sqrt(
        ((df['gt_latitude'] - df[f'{prefix}_latitude']) * 111000) ** 2 +
        ((df['gt_longitude'] - df[f'{prefix}_longitude']) * 111000) ** 2
    )
    if f'{prefix}_altitude' in df.columns and 'gt_altitude' in df.columns:
        df[f'{prefix}_alt_error_m'] = abs(df['gt_altitude'] - df[f'{prefix}_altitude'])

def merge_sensor_data(ground_truth, sensor_data):
    if ground_truth is None:
        print("Error: No ground truth loaded")
        return None

    merged = ground_truth.copy()

    # Rename GT columns
    column_mapping = {
        'latitude': 'gt_latitude',
        'longitude': 'gt_longitude',
        'altitude(m)': 'gt_altitude',
        'altitude': 'gt_altitude',
        'velocityX(mps)': 'gt_vel_x',
        'velocityY(mps)': 'gt_vel_y',
        'velocityZ(mps)': 'gt_vel_z',
        'speed(mps)': 'gt_speed'
    }
    for old, new in column_mapping.items():
        if old in merged.columns and new not in merged.columns:
            merged[new] = merged[old]

    if 'gt_altitude' not in merged.columns:
        merged['gt_altitude'] = 0

    merged = merged.sort_values('datetime(utc)').reset_index(drop=True)

    def safe_merge(sensor_name, df, rename_dict, tolerance='5s'):
        if df is None or df.empty:
            return

        df = df[df[list(rename_dict.keys())[0]].notna()].copy()
        df = df.rename(columns=rename_dict)

        use_cols = ['datetime(utc)'] + list(rename_dict.values())
        nonlocal merged
        merged = pd.merge_asof(
            merged.sort_values('datetime(utc)'),
            df.sort_values('datetime(utc)')[use_cols],
            on='datetime(utc)',
            direction='nearest',
            tolerance=pd.Timedelta(tolerance)
        )
        print(f"{sensor_name} data merged")

    # ALVIRA
    alvira = sensor_data.get('ALVIRA')
    if alvira is not None:
        alvira_cols = {
            'AlviraTracksTrackPosition_Latitude': 'alvira_latitude',
            'AlviraTracksTrackPosition_Longitude': 'alvira_longitude',
            'AlviraTracksTrackPosition_Altitude': 'alvira_altitude',
            'AlviraTracksTrackVelocity_Speed': 'alvira_speed',
            'AlviraTracksTrack_Classification': 'alvira_classification',
            'AlviraTracksTrack_Score': 'alvira_score'
        }
        safe_merge('ALVIRA', alvira, alvira_cols, '5s')
        if 'alvira_latitude' in merged.columns:
            calculate_positional_error(merged, 'alvira')

    # ARCUS
    arcus = sensor_data.get('ARCUS')
    if arcus is not None:
        arcus_cols = {
            'ArcusTracksTrackPosition_Latitude': 'arcus_latitude',
            'ArcusTracksTrackPosition_Longitude': 'arcus_longitude',
            'ArcusTracksTrackPosition_Altitude': 'arcus_altitude',
            'ArcusTracksTrackVelocity_Speed': 'arcus_speed',
            'ArcusTracksTrack_Classification': 'arcus_classification',
            'ArcusTracksTrack_Score': 'arcus_score'
        }
        safe_merge('ARCUS', arcus, arcus_cols, '5s')
        if 'arcus_latitude' in merged.columns:
            calculate_positional_error(merged, 'arcus')

    #DIANA
    diana = sensor_data.get('DIANA')
    if diana is not None:
        diana_cols = {
            'DianaTargetsTargetSignal_bearing_deg': 'diana_bearing',
            'DianaTargetsTargetSignal_range_m': 'diana_range',
            'DianaTargetsTargetSignal_snr_dB': 'diana_snr',
            'DianaTargetsTargetClassification_type': 'diana_classification',
            'DianaTargetsTargetClassification_score': 'diana_score'
        }
        safe_merge('DIANA', diana, diana_cols, '10s')

    #VENUS
    venus = sensor_data.get('VENUS')
    if venus is not None:
        venus_cols = {
            'VenusTrigger_Azimuth': 'venus_azimuth',
            'VenusTrigger_Frequency': 'venus_frequency',
            'VenusTriggerVenusName_isThreat': 'venus_threat_score'
        }
        safe_merge('VENUS', venus, venus_cols, '10s')

    merged = merged.sort_values('datetime(utc)').reset_index(drop=True)
    print(f"Data fusion complete: {len(merged)} records")
    return merged