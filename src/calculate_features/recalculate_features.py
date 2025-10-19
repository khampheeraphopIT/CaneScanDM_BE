import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api'))

import pandas as pd
from api.imageProcess.imageProcess import calculate_vegetation_indices, calculate_glcm_features, calculate_lbp_feature, calculate_edge_density
import cv2
import urllib.request
import numpy as np
import logging
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

df = pd.read_csv("dataset.csv")
logger.info(f"Loaded dataset with {len(df)} rows")

def extract_features(args):
    idx, image_url = args
    temp_file = f"temp_image_{idx}.jpg"
    try:
        # ดาวน์โหลดภาพ
        urllib.request.urlretrieve(image_url, temp_file)
        img = cv2.imread(temp_file)
        if img is None:
            logger.warning(f"Failed to load image: {image_url}")
            return idx, [0, 0, 0, 0, 0, 0, 0, 0]

        # ตรวจสอบว่าภาพเป็นสีดำทั้งหมดหรือไม่
        if np.mean(img) == 0:
            logger.warning(f"Image is completely black: {image_url}")
            os.remove(temp_file)
            return idx, [0, 0, 0, 0, 0, 0, 0, 0]

        # คำนวณ features
        indices = calculate_vegetation_indices(temp_file)
        vari = indices["vari"]
        exg = indices["exg"]
        cive = indices["cive"]

        glcm_features = calculate_glcm_features(temp_file)
        glcm_contrast = glcm_features["contrast"]
        glcm_homogeneity = glcm_features["homogeneity"]
        glcm_energy = glcm_features["energy"]

        lbp_feature = calculate_lbp_feature(temp_file)
        edge_density = calculate_edge_density(temp_file)

        # ลบไฟล์ temp_image หลังจากคำนวณเสร็จ
        os.remove(temp_file)

        return idx, [
    round(vari, 2), round(exg, 2), round(cive, 2),
    round(glcm_contrast, 2), round(glcm_homogeneity, 2), round(glcm_energy, 2),
    round(lbp_feature, 2), round(edge_density, 2)
]
    except Exception as e:
        logger.error(f"Error processing {image_url}: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return idx, [0, 0, 0, 0, 0, 0, 0, 0]

if __name__ == "__main__":
    urls = df['Image_URL'].tolist()
    tasks = [(idx, url) for idx, url in enumerate(urls)]

    with Pool(processes=4) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(extract_features, tasks), 1):
            logger.info(f"Processing image {i}/{len(tasks)}: {urls[result[0]]}")
            results.append(result)

    results.sort(key=lambda x: x[0])
    features_list = [result[1] for result in results]

    features_df = pd.DataFrame(features_list, columns=['VARI', 'ExG', 'CIVE', 'GLCM_Contrast', 'GLCM_Homogeneity', 'GLCM_Energy', 'LBP_Feature', 'Edge_Density'])
    df[['VARI', 'ExG', 'CIVE', 'GLCM_Contrast', 'GLCM_Homogeneity', 'GLCM_Energy', 'LBP_Feature', 'Edge_Density']] = features_df

    # ตรวจสอบการกระจายของ VARI
    logger.info(f"VARI statistics: min={df['VARI'].min()}, max={df['VARI'].max()}, mean={df['VARI'].mean()}")

    df.to_csv("dataset.csv", index=False)
    logger.info("Dataset updated with new features and saved to dataset.csv")