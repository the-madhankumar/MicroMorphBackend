import numpy as np
import cv2
import scipy
import mahotas
from skimage.feature import local_binary_pattern

class PolygonFeatureExtractor:
    """
    Compute 131+ contour/image features directly from polygon points (COCO format).
    """

    def __init__(self, polygon_points_list, orig_img=None):
        """
        polygon_points_list: list of polygons, each polygon is flat list [x1, y1, x2, y2, ...]
        orig_img: optional original BGR image for color and texture features
        """
        self.polygons = polygon_points_list
        self.orig_img = orig_img
        self.features = []

    @staticmethod
    def flatten_segmentation(seg):
        """
        Convert COCO segmentation (flat list) to OpenCV contour format Nx1x2
        """
        pts = np.array(seg).reshape(-1, 2)
        return pts.reshape(-1, 1, 2).astype(np.int32)

    def compute_features(self):
        """
        Compute features for all polygons.
        """
        for seg in self.polygons:
            contour = self.flatten_segmentation(seg)
            feat = self.compute_contour_properties(contour)
            self.features.append(feat)
        return self.features

    def compute_contour_properties(self, cnt):
        """
        Compute geometric, Hu, Zernike, color/gray, Haralick, LBP, Fourier descriptors
        for a single contour.
        """
        eps = 1e-12
        img = self.orig_img

        # ----------- GEOMETRIC PROPERTIES -----------
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        area_moment = M["m00"]
        cx = int(M["m10"]/M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"]/M["m00"]) if M["m00"] != 0 else 0
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area)/(perimeter**2) if perimeter > 0 else 0
        if len(cnt) >= 5:
            (xc, yc), (MA, ma), angle = cv2.fitEllipse(cnt)
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)
            a, b = major_axis/2, minor_axis/2
            eccentricity = np.sqrt(1 - (b*b)/(a*a)) if a>0 else 0
            roundness = minor_axis/major_axis if major_axis>0 else 0
        else:
            major_axis = minor_axis = eccentricity = roundness = 0
        equivalent_diameter = np.sqrt(4*area/np.pi) if area>0 else 0
        x,y,w,h = cv2.boundingRect(cnt)
        rectangularity = area/(w*h) if w*h>0 else 0
        shape_factor = (perimeter*perimeter)/area if area>0 else 0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        convexity = cv2.arcLength(hull, True)/perimeter if perimeter>0 else 0
        solidity = area/hull_area if hull_area>0 else 0
        extent = area/(w*h) if w*h>0 else 0

        # ----------- HU MOMENTS -----------
        hu = cv2.HuMoments(M).flatten()
        hu_log = -np.sign(hu)*np.log10(np.abs(hu)+eps)

        # ----------- ZERNIKE MOMENTS -----------
        if img is not None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        else:
            mask = np.zeros((100,100), dtype=np.uint8)
        cv2.drawContours(mask,[cnt],-1,255,-1)
        radius = min(mask.shape)//2
        zernike = mahotas.features.zernike_moments(mask, radius)

        # ----------- COLOR/GRAY FEATURES -----------
        if img is not None:
            img_bgr = img.copy()
        else:
            img_bgr = np.zeros((mask.shape[0],mask.shape[1],3), dtype=np.uint8)
        b,g,r = cv2.split(img_bgr)
        ratio_rg = np.mean(r)/(np.mean(g)+eps)
        ratio_rb = np.mean(r)/(np.mean(b)+eps)
        ratio_bg = np.mean(b)/(np.mean(g)+eps)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_mean = np.mean(gray)
        gray_std = np.std(gray)
        gray_flat = gray.reshape(-1)
        gray_skew = float(np.nan_to_num(scipy.stats.skew(gray_flat), nan=0.0))
        gray_kurt = float(np.nan_to_num(scipy.stats.kurtosis(gray_flat), nan=0.0))
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_norm = hist/(hist.sum()+eps)
        entropy = -np.sum(hist_norm*np.log2(hist_norm+eps))


        # ----------- HARALICK FEATURES -----------
        har = mahotas.features.haralick(gray, return_mean=True)

        # ----------- LBP (54 features) -----------
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        hist_lbp,_ = np.histogram(lbp.ravel(), bins=59, range=(0,59))
        hist_lbp = hist_lbp.astype(float)/(hist_lbp.sum()+eps)
        lbp_54 = hist_lbp[:54]

        # ----------- FOURIER DESCRIPTORS (10) -----------
        cnt_np = cnt.squeeze()
        if len(cnt_np.shape)==2 and cnt_np.shape[0]>20:
            complex_cnt = cnt_np[:,0] + 1j*cnt_np[:,1]
            fd = np.fft.fft(complex_cnt)
            fd_mag = np.abs(fd)
            fourier_10 = fd_mag[1:11]/(fd_mag[1]+eps)
        else:
            fourier_10 = np.zeros(10)

        feature = {
            "area": area,
            "centroid": (cx,cy),
            "perimeter": perimeter,
            "circularity": circularity,
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "eccentricity": eccentricity,
            "equivalent_diameter": equivalent_diameter,
            "rectangularity": rectangularity,
            "roundness": roundness,
            "shape_factor": shape_factor,
            "convexity": convexity,
            "solidity": solidity,
            "extent": extent,
            **{f"hu_{i+1}": hu_log[i] for i in range(7)},
            **{f"z_{i}": zernike[i] for i in range(25)},
            "ratio_rg": ratio_rg,
            "ratio_rb": ratio_rb,
            "ratio_bg": ratio_bg,
            "gray_mean": gray_mean,
            "gray_std": gray_std,
            "gray_skew": gray_skew,
            "gray_kurt": gray_kurt,
            "gray_entropy": entropy,
            **{f"h_{i}": har[i] for i in range(13)},
            **{f"lbp_{i}": lbp_54[i] for i in range(54)},
            **{f"fourier_{i}": fourier_10[i] for i in range(10)},
        }
        return feature