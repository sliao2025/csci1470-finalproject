import os
import shutil
import pandas as pd

fma_audio_path = "./fma_medium"
tracks_csv_path = "./tracks_metadata.csv"
df = pd.read_csv(tracks_csv_path)
df.set_index('track_id', inplace=True)

def get_genre(track_id):
    try:
        genre = df.loc[track_id, 'genre']  # Replace '41st_column_name' with the actual name of the 41st column
        return genre
    except KeyError:
        return "Track ID not found"
    
genre = df.loc["2", 'genre']
print(genre)

new_folder_path = "./renamed_fma_audio"
os.makedirs(new_folder_path)

i = 0
for folder in os.listdir(fma_audio_path):
    folder_path = os.path.join(fma_audio_path, folder)
    # Check if the item is not a .DS_Store file
    if folder_path.endswith('.DS_Store'): continue
    if not os.path.isdir(folder_path): continue 
    # Iterate over all files in the directory
    for filename in os.listdir(folder_path):
        # Check if the file is actually a file (not a directory)
        if os.path.isfile(os.path.join(folder_path, filename)):
            genre = get_genre(filename[:filename.rfind(".")].lstrip('0') or '0').replace("/", "_")
            # if the genre exists for a given track
            if not pd.isna(genre):
                # Construct the new filename
                new_filename = f"{i}_{genre}.mp3"
                # Rename the file
                os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
                # Move file outside of folder 
                shutil.copy(os.path.join(folder_path, new_filename), new_folder_path)
                i += 1