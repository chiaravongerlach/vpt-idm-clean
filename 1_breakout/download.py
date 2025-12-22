import requests
import time
#download data 
# Zenodo API endpoint
record_id = "3451402"
api_url = f"https://zenodo.org/api/records/{record_id}"

print("Fetching file list from Zenodo...")
time.sleep(2)

response = requests.get(api_url)
if response.status_code == 200:
    data = response.json()
    files = data.get('files', [])
    
    # Find breakout file
    breakout_file = None
    for f in files:
        if 'breakout' in f['key'].lower():
            breakout_file = f
            break
    
    if breakout_file:
        print(f"Found: {breakout_file['key']}")
        print(f"Size: {breakout_file['size'] / (1024**3):.2f} GB")
        download_url = breakout_file['links']['self']
        
        print(f"Downloading from: {download_url}")
        
        # Download with proper streaming
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(breakout_file['key'], 'wb') as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024**2)
                            mb_total = total_size / (1024**2)
                            print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
        print("\nDownload complete!")
        print(f"File saved as: {breakout_file['key']}")
    else:
        print("Breakout file not found.")
else:
    print(f"Error: {response.status_code}")
