import pandas as pd
import numpy as np
from pathlib import Path
import os
import warnings

warnings.filterwarnings('ignore')

class DataIngestion:
    def __init__(self, base_data_path="data/apt_datasets"):
        self.base_data_path = Path(base_data_path)
        self.base_data_path.mkdir(parents=True, exist_ok=True)

    def _check_and_download_instructions(self, dataset_name, expected_files, download_url=None):
        dataset_path = self.base_data_path / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)

        missing_files = []
        for f in expected_files:
            if not (dataset_path / f).exists():
                missing_files.append(f)

        if missing_files:
            print(f"\n⚠️  Missing files for {dataset_name}: {missing_files}")
            print(f"   Please ensure these files are placed in: {dataset_path}")
            if download_url:
                print(f"   You might be able to download them from: {download_url}")
            return False
        return True

    def convert_csv_to_parquet(self, input_csv_path, output_parquet_path, chunksize=100000):
        print(f"Converting {input_csv_path} to Parquet...")
        try:
            # Read CSV in chunks and append to Parquet
            for i, chunk in enumerate(pd.read_csv(input_csv_path, chunksize=chunksize)):
                if i == 0:
                    chunk.to_parquet(output_parquet_path, index=False, mode='w')
                else:
                    chunk.to_parquet(output_parquet_path, index=False, mode='a')
                print(f"  Processed chunk {i+1}...")
            print(f"Successfully converted {input_csv_path} to {output_parquet_path}")
            return True
        except Exception as e:
            print(f"Error converting CSV to Parquet: {e}")
            return False

    def load_parquet_chunks(self, parquet_path, chunk_size=100000):
        print(f"Loading Parquet file in chunks: {parquet_path}")
        try:
            # This is a simplified chunking for demonstration.
            # For true memory efficiency with large files, consider iterating over row groups
            # or using dask/pyarrow for more advanced chunking.
            df = pd.read_parquet(parquet_path)
            num_chunks = (len(df) + chunk_size - 1) // chunk_size
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(df))
                yield df.iloc[start_idx:end_idx]
        except Exception as e:
            print(f"Error loading Parquet chunks: {e}")
            yield None

    def get_unsw_nb15_data(self):
        dataset_name = "UNSW-NB15"
        # Assuming the training and testing sets are the primary files
        expected_parquet_files = [
            "UNSW_NB15_training-set.parquet",
            "UNSW_NB15_testing-set.parquet"
        ]
        
        # Check for raw CSVs if parquet files are not present
        expected_csv_files = [
            "UNSW_NB15_training-set.csv",
            "UNSW_NB15_testing-set.csv"
        ]

        # Check if Parquet files exist
        if self._check_and_download_instructions(dataset_name, expected_parquet_files):
            print(f"✅ UNSW-NB15 Parquet data found in {self.base_data_path / dataset_name}")
            return {
                "training": self.base_data_path / dataset_name / expected_parquet_files[0],
                "testing": self.base_data_path / dataset_name / expected_parquet_files[1]
            }
        else:
            print(f"Attempting to convert UNSW-NB15 CSVs to Parquet...")
            # Check for CSVs and convert
            if self._check_and_download_instructions(dataset_name, expected_csv_files, 
                                                      download_url="https://www.unsw.adfa.edu.au/unsw-nb15-dataset/"):
                train_csv = self.base_data_path / dataset_name / expected_csv_files[0]
                test_csv = self.base_data_path / dataset_name / expected_csv_files[1]
                train_parquet = self.base_data_path / dataset_name / expected_parquet_files[0]
                test_parquet = self.base_data_path / dataset_name / expected_parquet_files[1]

                if self.convert_csv_to_parquet(train_csv, train_parquet) and \
                   self.convert_csv_to_parquet(test_csv, test_parquet):
                    return {
                        "training": train_parquet,
                        "testing": test_parquet
                    }
            print("❌ UNSW-NB15 data (CSV or Parquet) not found. Please download manually.")
            return None

    def get_cic_ids_2017_data(self):
        dataset_name = "cicids2017"
        # Based on preprocess_cicids2017.py
        expected_csv_files = [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv", 
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        ]
        expected_parquet_files = [f.replace('.csv', '.parquet') for f in expected_csv_files]

        # Check if Parquet files exist
        if self._check_and_download_instructions(dataset_name, expected_parquet_files):
            print(f"✅ CIC-IDS-2017 Parquet data found in {self.base_data_path / dataset_name}")
            return {f.replace('.parquet', ''): self.base_data_path / dataset_name / f for f in expected_parquet_files}
        else:
            print(f"Attempting to convert CIC-IDS-2017 CSVs to Parquet...")
            if self._check_and_download_instructions(dataset_name, expected_csv_files, 
                                                      download_url="https://www.unb.ca/cic/datasets/ids-2017.html"):
                parquet_paths = {}
                all_converted = True
                for csv_file in expected_csv_files:
                    csv_path = self.base_data_path / dataset_name / csv_file
                    parquet_path = self.base_data_path / dataset_name / csv_file.replace('.csv', '.parquet')
                    if not self.convert_csv_to_parquet(csv_path, parquet_path):
                        all_converted = False
                        break
                    parquet_paths[csv_file.replace('.csv', '')] = parquet_path
                if all_converted:
                    return parquet_paths
            print("❌ CIC-IDS-2017 data (CSV or Parquet) not found. Please download manually.")
            return None

    def get_apt_netflow_data(self):
        dataset_name = "apt_netflow"
        # This is a placeholder. Actual APT/Netflow data might be PCAP or other formats.
        # For now, assume a single large CSV that needs conversion.
        expected_csv_files = ["apt_netflow_data.csv"]
        expected_parquet_files = ["apt_netflow_data.parquet"]

        if self._check_and_download_instructions(dataset_name, expected_parquet_files):
            print(f"✅ APT/Netflow Parquet data found in {self.base_data_path / dataset_name}")
            return self.base_data_path / dataset_name / expected_parquet_files[0]
        else:
            print(f"Attempting to convert APT/Netflow CSV to Parquet...")
            if self._check_and_download_instructions(dataset_name, expected_csv_files, 
                                                      download_url="[Insert APT/Netflow Data Download URL Here]"):
                csv_path = self.base_data_path / dataset_name / expected_csv_files[0]
                parquet_path = self.base_data_path / dataset_name / expected_parquet_files[0]
                if self.convert_csv_to_parquet(csv_path, parquet_path):
                    return parquet_path
            print("❌ APT/Netflow data (CSV or Parquet) not found. Please download manually.")
            return None

if __name__ == "__main__":
    ingestion = DataIngestion()

    print("\n--- Getting UNSW-NB15 Data ---")
    unsw_paths = ingestion.get_unsw_nb15_data()
    if unsw_paths:
        print(f"UNSW-NB15 data paths: {unsw_paths}")
        # Example of loading in chunks
        # for chunk in ingestion.load_parquet_chunks(unsw_paths['training']):
        #     if chunk is not None:
        #         print(f"  Loaded UNSW-NB15 training chunk of shape: {chunk.shape}")

    print("\n--- Getting CIC-IDS-2017 Data ---")
    cic_paths = ingestion.get_cic_ids_2017_data()
    if cic_paths:
        print(f"CIC-IDS-2017 data paths: {cic_paths}")
        # Example of loading in chunks
        # for day, path in cic_paths.items():
        #     for chunk in ingestion.load_parquet_chunks(path):
        #         if chunk is not None:
        #             print(f"  Loaded CIC-IDS-2017 {day} chunk of shape: {chunk.shape}")

    print("\n--- Getting APT/Netflow Data ---")
    apt_netflow_path = ingestion.get_apt_netflow_data()
    if apt_netflow_path:
        print(f"APT/Netflow data path: {apt_netflow_path}")
        # Example of loading in chunks
        # for chunk in ingestion.load_parquet_chunks(apt_netflow_path):
        #     if chunk is not None:
        #         print(f"  Loaded APT/Netflow chunk of shape: {chunk.shape}")
