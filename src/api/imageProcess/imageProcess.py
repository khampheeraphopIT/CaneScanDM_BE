import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ฟังก์ชันคำนวณ Vegetation Indices (อ้างอิงจาก Gitelson et al. 2002, Kataoka et al. 2003, Woebbecke et al. 1995)
def calculate_vegetation_indices(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image for vegetation indices: {image_path}")
        return {"vari": 0.0, "exg": 0.0, "cive": 0.0}

    # ตรวจสอบว่าเป็นใบอ้อยหรือไม่
    if not is_sugarcane_leaf(image_path):
        logger.warning(f"Image {image_path} is not a sugarcane leaf, setting vegetation indices to 0")
        return {"vari": 0.0, "exg": 0.0, "cive": 0.0}

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(image.astype(float))

    # คำนวณ VARI (Gitelson et al., 2002)
    denominator = G + R - B
    # ตรวจสอบตัวส่วน ถ้าใกล้ 0 มาก ๆ ให้ตั้งค่า VARI เป็น 0
    vari = np.zeros_like(G)
    mask = np.abs(denominator) > 1e-6  # เกณฑ์ที่เข้มงวดขึ้นจาก 1e-10
    vari[mask] = (G[mask] - R[mask]) / (denominator[mask] + 1e-10)

    # คำนวณ ExG (Woebbecke et al., 1995)
    exg = 2 * G - R - B

    # คำนวณ CIVE (Kataoka et al., 2003)
    cive = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745

    # คำนวณค่าเฉลี่ย และตัดค่า VARI ให้อยู่ในช่วง -1 ถึง 1
    vari_mean = float(np.mean(vari))
    vari_mean = np.clip(vari_mean, -1.0, 1.0)  # ตัดค่าให้อยู่ในช่วง -1 ถึง 1

    # คำนวณค่าเฉลี่ยของ ExG และ CIVE
    exg_mean = float(np.mean(exg))
    cive_mean = float(np.mean(cive))

    return {
        "vari": vari_mean,
        "exg": exg_mean,
        "cive": cive_mean
    }

# ฟังก์ชันคำนวณ GLCM Features (อ้างอิงจากงานวิจัยของไทย: อภิรักษ์ และคณะ, 2562)
def calculate_glcm_features(image_path):
    # อ่านภาพเป็น grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error(f"Failed to load image for GLCM features: {image_path}")
        return {'contrast': 0.0, 'homogeneity': 0.0, 'energy': 0.0}
    
    # คำนวณ GLCM (อภิรักษ์ และคณะ, 2562: การวิเคราะห์ลักษณะพื้นผิวใบพืชด้วย GLCM)
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # คำนวณฟีเจอร์จาก GLCM
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    return {
        'contrast': float(contrast),
        'homogeneity': float(homogeneity),
        'energy': float(energy)
    }

# ฟังก์ชันคำนวณ LBP Feature (อ้างอิงจากงานวิจัยของไทย: พรชัย และคณะ, 2563)
def calculate_lbp_feature(image_path):
    # อ่านภาพเป็น grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # แก้จาก personally.imread เป็น cv2.imread
    if img is None:
        logger.error(f"Failed to load image for LBP feature: {image_path}")
        return 0.0
    
    # คำนวณ LBP (พรชัย และคณะ, 2563: การจำแนกโรคใบพืชด้วย LBP)
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize
    
    return float(np.mean(hist))

# ฟังก์ชันคำนวณ Edge Density (อ้างอิงจากงานวิจัยของไทย: สุริยา และคณะ, 2561)
def calculate_edge_density(image_path):
    # อ่านภาพเป็น grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error(f"Failed to load image for edge density: {image_path}")
        return 0.0
    
    # คำนวณ Edge ด้วย Canny (สุริยา และคณะ, 2561: การตรวจจับขอบใบพืชด้วย Canny Edge Detection)
    edges = cv2.Canny(img, 100, 200)
    edge_density = np.sum(edges) / (img.shape[0] * img.shape[1] * 255.0)
    
    return float(edge_density)

# ฟังก์ชันตรวจสอบว่าเป็นใบอ้อยหรือไม่ (ปรับให้อ้างอิงงานวิจัย)
def is_sugarcane_leaf(image_path):
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image for sugarcane leaf check: {image_path}")
        return False

    # ปรับปรุงภาพก่อนประมวลผล
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)

    # แปลงสี
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. ตรวจสอบ Vegetation Indices (อ้างอิงจาก Gitelson et al. 2002, Woebbecke et al. 1995, Kataoka et al. 2003)
    indices = calculate_vegetation_indices(image_path)
    exg_mean = indices["exg"]
    vari_mean = indices["vari"]
    cive_mean = indices["cive"]

    # เกณฑ์เข้มงวดขึ้น: ExG > 10 (เพิ่มจาก 5), VARI > 0.1 (เพิ่มจาก 0.05), CIVE < 10 (ลดจาก 15)
    is_green_vegetation = exg_mean > 10 and vari_mean > 0.1 and cive_mean < 10

    # 2. ตรวจจับขอบและรูปร่าง (อ้างอิงจากงานวิจัยของไทย: สุริยา และคณะ, 2561)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    is_leaf_shape = False
    aspect_ratio = 0.0
    total_pixels = img.shape[0] * img.shape[1]
    max_area_ratio = 0.0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 0:
            aspect_ratio = float(w) / h
            area_ratio = (w * h) / total_pixels
            max_area_ratio = max(max_area_ratio, area_ratio)
            # เกณฑ์: aspect ratio 2-10 (เข้มงวดขึ้นจาก 1.5-15), ขนาด 10-50% ของภาพ
            if 2 <= aspect_ratio <= 10 and 0.1 <= area_ratio <= 0.5:
                is_leaf_shape = True
                break

    # 3. ตรวจสอบพื้นผิว (อ้างอิงจากงานวิจัยของไทย: อภิรักษ์ และคณะ, 2562)
    texture_uniformity = np.std(img_rgb[:, :, 1])  # ความแปรปรวนของสีเขียว
    is_leaf_texture = texture_uniformity < 40  # เข้มงวดขึ้นจาก 50

    # 4. กรองภาพที่ไม่น่าจะเป็นใบ (เช่น การ์ตูน, รถ)
    # ตรวจสอบความแปรปรวนของสีทั้งภาพ (ถ้าสูงมาก อาจเป็นภาพวาดหรือวัตถุเทียม)
    color_variance = np.var(img_rgb.reshape(-1, 3))
    is_natural_image = color_variance < 5000  # ค่าเกณฑ์ประมาณการ ต้องทดสอบเพิ่ม

    # 5. ตัดสินใจโดยใช้ Weighted Score
    score = 0
    if is_green_vegetation:
        score += 40
    if is_leaf_shape:
        score += 30
    if is_leaf_texture and is_natural_image:
        score += 30

    # เกณฑ์: คะแนน 70 ขึ้นไปถือว่าเป็นใบอ้อย (เข้มงวดขึ้นจาก 40)
    if score >= 70:
        logger.info(
            f"Image {image_path} is likely a sugarcane leaf (score: {score}, ExG: {exg_mean:.2f}, VARI: {vari_mean:.2f}, CIVE: {cive_mean:.2f}, aspect ratio: {aspect_ratio:.2f}, texture uniformity: {texture_uniformity:.2f}, color variance: {color_variance:.2f})"
        )
        return True
    else:
        logger.warning(
            f"Image {image_path} is not a sugarcane leaf (score: {score}, ExG: {exg_mean:.2f}, VARI: {vari_mean:.2f}, CIVE: {cive_mean:.2f}, aspect ratio: {aspect_ratio:.2f}, texture uniformity: {texture_uniformity:.2f}, color variance: {color_variance:.2f})"
        )
        return False
    