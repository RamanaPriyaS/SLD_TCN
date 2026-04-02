import os
import glob
import numpy as np

def augment_row(row):
    thumb_tip_x, thumb_tip_y, thumb_tip_z = row[12], row[13], row[14]
    
    target_indices = [5, 6, 9, 10, 13, 14, 17, 18]
    distances = []
    
    for idx in target_indices:
        base_idx = idx * 3
        tx, ty, tz = row[base_idx], row[base_idx+1], row[base_idx+2]
        dist = np.sqrt((tx - thumb_tip_x)**2 + (ty - thumb_tip_y)**2 + (tz - thumb_tip_z)**2)
        distances.append(dist)
        
    return np.concatenate([row, np.array(distances, dtype=row.dtype)])

def main():
    source_dir = 'data/alphabets_npy'
    target_dir = 'data/alphabets_augmented_npy'
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = glob.glob(os.path.join(source_dir, '*.npy'))
    print(f"Found {len(files)} files to process in {source_dir}.")
    
    processed = 0
    skipped = 0
    
    for f in files:
        try:
            data = np.load(f)
            
            # Allow reprocessing the same file if someone accidentally runs it twice, 
            # though here we are assuming raw data has shape 63
            if len(data.shape) == 1:
                if len(data) == 63:
                    new_data = augment_row(data)
                elif len(data) == 71:
                    new_data = data # Already augmented
                else:
                    print(f"Skipping {f}: abnormal shape {data.shape}")
                    skipped += 1
                    continue
            elif len(data.shape) == 2:
                if data.shape[1] == 63:
                    new_data = np.array([augment_row(row) for row in data])
                elif data.shape[-1] == 71:
                    new_data = data # Already augmented
                else:
                    print(f"Skipping {f}: abnormal shape {data.shape}")
                    skipped += 1
                    continue
            else:
                print(f"Skipping {f}: abnormal dimensions {data.shape}")
                skipped += 1
                continue
                
            basename = os.path.basename(f)
            out_path = os.path.join(target_dir, basename)
            np.save(out_path, new_data)
            processed += 1
            
        except Exception as e:
            print(f"Error processing {f}: {e}")
            skipped += 1
            
    print(f"Successfully processed {processed} files into {target_dir}.")
    print(f"Skipped {skipped} files.")

if __name__ == '__main__':
    main()
