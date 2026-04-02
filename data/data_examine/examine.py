import os
import glob
from collections import Counter
import pandas as pd
import json

def examine_data(data_dir="../data/processed", output_dir="."):
    """
    Examine the processed data directory and count the number of valid sequences per word.
    """
    print(f"Examining data in {data_dir}...")
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(data_dir, "*.npy"))
    print(f"Found a total of {len(files)} processed sequences.")
    
    # Extract the class name from each file (assuming format class_seqid.npy)
    classes = [os.path.basename(f).split('_')[0] for f in files]
    
    # Count the occurrences of each class
    class_counts = Counter(classes)
    
    # Convert to a pandas DataFrame for nice formatting and easier exporting
    df = pd.DataFrame.from_dict(class_counts, orient='index', columns=['Count']).reset_index()
    df = df.rename(columns={'index': 'Word'})
    df = df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    
    print("\nData Distribution:")
    print("-" * 30)
    print(df.to_string(index=False))
    print("-" * 30)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "data_distribution.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved distribution text report to {csv_path}")
    
    # Save to JSON
    json_path = os.path.join(output_dir, "data_distribution.json")
    with open(json_path, 'w') as f:
        json.dump(class_counts, f, indent=4)
    print(f"Saved distribution JSON report to {json_path}")
    
if __name__ == "__main__":
    # Ensure this runs correctly regardless of where it's called from
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    examine_data(data_dir=data_dir, output_dir=output_dir)
