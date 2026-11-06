import csv
import os
from time import time

class CSVLogger:
    def __init__(self, config):
        """
        Initializes the logger.
        
        Args:
            filepath (str): Path to the output CSV file.
        """
        os.makedirs("run_logs", exist_ok=True)
        self.filepath = f"run_logs/{config['run_name']}.csv"
        self.headers = []
        overwrite = config["overwrite_logs"]
        self.start_time = time()

        # If overwrite is requested and file exists, remove it to start fresh
        if overwrite and os.path.exists(self.filepath):
            os.remove(self.filepath)
        
        # Check if file already exists to load existing headers
        # This allows resuming an experiment without overwriting the header logic
        if not overwrite and os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
            with open(self.filepath, 'r', newline='') as f:
                reader = csv.reader(f)
                try:
                    self.headers = next(reader)
                except StopIteration:
                    pass

    def log(self, row_dict):
        """
        Logs a dictionary row to the CSV. Adjusts headers if new keys are found.
        
        Args:
            row_dict (dict): Dictionary of headers-to-values (e.g., {'epoch': 1, 'loss': 0.5})
        """
        row_dict["elapsed_time"] = time() - self.start_time
        # 1. Identify if there are any new keys in the input that aren't in our headers
        new_keys = [key for key in row_dict.keys() if key not in self.headers]

        # 2. If new keys are found, we must update the CSV file structure
        if new_keys:
            self.headers.extend(new_keys)
            self._rewrite_file_with_new_headers()
            
        # 3. Append the new row
        # We assume the file is now prepared with the correct headers
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            
            # If this is the very first write (file is empty), write the header
            if f.tell() == 0:
                writer.writeheader()
                
            writer.writerow(row_dict)

    def _rewrite_file_with_new_headers(self):
        """
        Reads the existing file, updates the headers, and rewrites the content.
        Previous rows will have empty values for the new columns.
        """
        rows = []
        if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
            with open(self.filepath, 'r', newline='') as f:
                # Use DictReader to map old data to old headers automatically
                reader = csv.DictReader(f)
                rows = list(reader)

        # Rewrite the file from scratch with the new superset of headers
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            # DictWriter handles missing keys by leaving them empty (default behavior)
            writer.writerows(rows)