import pandas as pd
import requests
import os
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ลบโฟลเดอร์ images เดิม (ถ้ามี) และสร้างใหม่
if os.path.exists("images"):
    shutil.rmtree("images")
os.makedirs("images", exist_ok=True)
logger.info("สร้างโฟลเดอร์ images ใหม่เรียบร้อย")

# อ่าน dataset.csv
df = pd.read_csv("dataset.csv")
logger.info(f"อ่าน dataset.csv เสร็จแล้ว มี {len(df)} แถว")

# ฟังก์ชันแก้ไข URL ให้ถูกต้อง
def fix_google_drive_url(url):
    try:
        file_id = url.split("id=")[1].split("&")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    except Exception as e:
        logger.error(f"Failed to fix URL {url}: {e}")
        return None

# ฟังก์ชันดาวน์โหลดและบันทึกภาพ
def download_and_save_image(idx_url):
    idx, url = idx_url
    image_name = f"image_{idx:05d}.jpg"
    save_path = os.path.join("images", image_name)
    
    # ถ้าไฟล์มีอยู่แล้ว ข้ามไป
    if os.path.exists(save_path):
        return idx, save_path

    # แก้ไข URL
    fixed_url = fix_google_drive_url(url)
    if not fixed_url:
        return idx, save_path

    try:
        response = requests.get(fixed_url, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            logger.warning(f"URL {fixed_url} does not point to an image (Content-Type: {content_type})")
            return idx, save_path
        with open(save_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded {fixed_url} to {save_path}")
        return idx, save_path
    except Exception as e:
        logger.error(f"Failed to download {fixed_url}: {e}")
        return idx, save_path

# ดาวน์โหลดภาพแบบ multi-thread เพื่อให้เร็วขึ้น
def download_images_concurrently(image_urls):
    new_image_paths = [None] * len(image_urls)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_and_save_image, (idx, url)) for idx, url in enumerate(image_urls)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            idx, save_path = future.result()
            new_image_paths[idx] = save_path
    return new_image_paths

# ดาวน์โหลดภาพทั้งหมด
image_urls = df['Image_URL'].values
new_image_paths = download_images_concurrently(image_urls)

# อัปเดต DataFrame
df['Image_URL'] = new_image_paths

# บันทึก dataset ใหม่
df.to_csv("dataset_updated.csv", index=False)
logger.info("Updated dataset saved as dataset_updated.csv")