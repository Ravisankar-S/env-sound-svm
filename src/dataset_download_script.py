import os, zipfile, requests
from tqdm import tqdm

ESC_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

def download_and_extract(dest="data/raw"):
    os.makedirs(dest, exist_ok=True)
    zip_path = "ESC-50-master.zip"

    print("⬇️  Downloading ESC-50 (includes ESC-10 subset)...")
    r = requests.get(ESC_URL, stream=True)
    r.raise_for_status()

    total_size = int(r.headers.get('content-length', 0))
    block_size = 8192

    with open(zip_path, "wb") as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in r.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")
    os.remove(zip_path)

    src_dir = "ESC-50-master/audio"
    meta_file = "ESC-50-master/meta/esc50.csv"

    import pandas as pd
    df = pd.read_csv(meta_file)
    
    esc10_df = df[df['esc10'] == True]
    final_labels = esc10_df['category'].unique()

    print(f"Using ESC-10 classes: {final_labels}")

    for label in final_labels:
        os.makedirs(os.path.join(dest, label), exist_ok=True)

    for _, row in esc10_df.iterrows():
        src = os.path.join(src_dir, row.filename)
        dst = os.path.join(dest, row.category, row.filename)
        if os.path.exists(src):
            os.rename(src, dst)

    import shutil
    if os.path.exists("ESC-50-master"):
        shutil.rmtree("ESC-50-master")
    print("✅ ESC-10 subset ready in data/raw/")

if __name__ == "__main__":
    download_and_extract()