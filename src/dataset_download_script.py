import os, zipfile, requests
import shutil

ESC_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

def download_and_extract(dest="data/raw"):
    os.makedirs(dest, exist_ok=True)
    zip_path = "ESC-50-master.zip"

    print(" Downloading ESC-50 (includes ESC-10 subset)...")
    r = requests.get(ESC_URL, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(" Download complete.")

    print(" Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")
    os.remove(zip_path)

    src_dir = "ESC-50-master/audio"
    meta_file = "ESC-50-master/meta/esc50.csv"

    import pandas as pd
    df = pd.read_csv(meta_file)
    esc10_labels = [
        "dog", "rain", "sea_waves", "baby_cry", "clock_tick",
        "chainsaw", "crackling_fire", "helicopter", "rooster", "crying_baby"
    ]
    esc10_df = df[df.category.isin(esc10_labels)]

    for label in esc10_labels:
        os.makedirs(os.path.join(dest, label), exist_ok=True)

    for _, row in esc10_df.iterrows():
        src = os.path.join(src_dir, row.filename)
        dst = os.path.join(dest, row.category, row.filename)
        os.rename(src, dst)

    shutil.rmtree("ESC-50-master")
    print("ESC-10 subset ready in data/raw/")

if __name__ == "__main__":
    download_and_extract()
