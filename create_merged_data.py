import os
from multi_processor import MultiSensorDataProcessor

def main():
    data_dir = os.path.abspath("data/src/icmcis-drone-detection/train/train/Scenario_1_1")  # folder where raw CSV files are
    output_dir = os.path.abspath("data/processed")      # output folder for merged data
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "merged_data.csv")

    processor = MultiSensorDataProcessor(data_dir)

    if not processor.load_all_data():
        print("Failed to load all sensor data. Exiting.")
        return

    merged = processor.merge_sensor_data()
    if merged is None:
        print("Failed to merge sensor data. Exiting.")
        return

    print(f"Saving merged data to: {output_file}")
    merged.to_csv(output_file, index=False)
    print("Merged data saved successfully.")

if __name__ == "__main__":
    main()