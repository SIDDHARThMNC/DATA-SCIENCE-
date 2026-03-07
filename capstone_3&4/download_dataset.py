"""
Helper script to download dataset from GitHub
Dataset: DatasetCapstoneProject3
Source: https://github.com/himanshusar123/Datasets
"""

import urllib.request
import os

def download_dataset():
    """Download marketing dataset from GitHub"""
    
    print("=" * 70)
    print("DATASET DOWNLOAD HELPER")
    print("=" * 70)
    
    # GitHub raw URL for the dataset
    # Note: Update this URL with the actual file path from the repository
    base_url = "https://raw.githubusercontent.com/himanshusar123/Datasets/main/"
    
    # Possible dataset filenames
    possible_files = [
        "DatasetCapstoneProject3/marketing_data.csv",
        "DatasetCapstoneProject3.csv",
        "marketing_data.csv",
        "capstone3_marketing.csv"
    ]
    
    print("\n📥 Attempting to download dataset...")
    print(f"Source: {base_url}")
    
    success = False
    for filename in possible_files:
        try:
            url = base_url + filename
            output_path = "capstone_3&4/marketing_data.csv"
            
            print(f"\nTrying: {url}")
            urllib.request.urlretrieve(url, output_path)
            
            # Check if file was downloaded
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✓ Dataset downloaded successfully!")
                print(f"✓ Saved to: {output_path}")
                success = True
                break
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            continue
    
    if not success:
        print("\n" + "=" * 70)
        print("⚠️  MANUAL DOWNLOAD REQUIRED")
        print("=" * 70)
        print("\nPlease download the dataset manually:")
        print("1. Visit: https://github.com/himanshusar123/Datasets")
        print("2. Navigate to: DatasetCapstoneProject3")
        print("3. Download the CSV file")
        print("4. Save it as: capstone_3&4/marketing_data.csv")
        print("\nThen run: python capstone_3_smart_marketing_prediction.py")
    else:
        print("\n" + "=" * 70)
        print("✓ READY TO RUN!")
        print("=" * 70)
        print("\nRun the main script:")
        print("python capstone_3_smart_marketing_prediction.py")

if __name__ == "__main__":
    download_dataset()
