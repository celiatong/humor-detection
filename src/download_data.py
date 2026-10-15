import os
import urllib.request

# Define direct S3 download links
urls = {
    "Humorous.csv": "https://humor-detection-pds.s3-us-west-2.amazonaws.com/Humorous.csv",
    "Non-humorous-unbiased.csv": "https://humor-detection-pds.s3-us-west-2.amazonaws.com/Non-humorous-unbiased.csv",
    "Non-humorous-biased.csv": "https://humor-detection-pds.s3-us-west-2.amazonaws.com/Non-humours-biased.csv", # Note the typo 'Non-humours' in AWS's actual path
    "README.txt": "https://humor-detection-pds.s3-us-west-2.amazonaws.com/README.txt"
}

dest_dir = "/Users/sisi/Desktop/Masters_degree/5018/humor-detection/data"
os.makedirs(dest_dir, exist_ok=True)

print("Starting dataset downloads from AWS Open Data Registry...")
for filename, url in urls.items():
    dest_path = os.path.join(dest_dir, filename)
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Successfully saved to {dest_path}")
    except Exception as e:
        print(f"Failed to download {filename}. Error: {e}")

print("\nDownload process complete!")
