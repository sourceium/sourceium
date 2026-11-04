import os
import struct
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage import measure

def run_depth_anything(image_bgr, model_path="models/depth_anything_v2_vits.onnx"):
    """Runs Depth Anything V2 ONNX with fallback."""
    if os.path.exists(model_path):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            h, w = image_bgr.shape[:2]
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            norm = (rgb - mean) / std
            blob = cv2.resize(norm, (518, 518)).transpose(2, 0, 1)[np.newaxis, ...]
            out = session.run(None, {session.get_inputs()[0].name: blob})[0]
            depth = cv2.resize(out[0, 0], (w, h))
            return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        except Exception as e:
            print(f"ONNX inference warning: {e}. Falling back to edge map.")
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (21, 21), 0) / 255.0

def process_clay_mask(img_gray):
    """Cut out clay: grey below 226, strip floor shadows, fill holes."""
    binary = (img_gray < 226).astype(np.uint8)
    binary[int(binary.shape[0] * 0.94):, :] = 0  # strip ground contact shadow
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_idx).astype(np.uint8)
    else:
        mask = binary
    
    return binary_fill_holes(mask).astype(np.uint8)

def cut_out_portrait(photo_bgr):
    """Cut out portrait with soft hair alpha and closed neck/collar gaps."""
    hsv = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]
    
    # Foreground separation on black studio backdrop
    _, raw_mask = cv2.threshold(val, 16, 255, cv2.THRESH_BINARY)
    
    # Close neck and collar gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    closed = binary_fill_holes(closed).astype(np.uint8) * 255
    
    # Soft alpha feathering for flyaway hair edges
    alpha = cv2.GaussianBlur(closed.astype(np.float32), (7, 7), 0) / 255.0
    
    b, g, r = cv2.split(photo_bgr)
    rgba = cv2.merge([b, g, r, (alpha * 255).astype(np.uint8)])
    return rgba

def carve_voxel_hull(m_front, m_side, pitch=2.5):
    """Carves 3D volume matching silhouettes with smooth elliptical cross-sections."""
    h, wf = m_front.shape
    _, ws = m_side.shape
    
    vx, vy, vz = int(wf / pitch), int(h / pitch), int(ws / pitch)
    
    f_res = cv2.resize(m_front, (vx, vy), interpolation=cv2.INTER_NEAREST) > 0
    s_res = cv2.resize(m_side, (vz, vy), interpolation=cv2.INTER_NEAREST) > 0
    
    grid = np.zeros((vx, vy, vz), dtype=np.float32)
    x_mesh, z_mesh = np.meshgrid(np.arange(vx), np.arange(vz), indexing='ij')
    
    for y in range(vy):
        x_pts = np.where(f_res[y, :])[0]
        z_pts = np.where(s_res[y, :])[0]
        if len(x_pts) == 0 or len(z_pts) == 0:
            continue
        
        cx, rx = (x_pts[0] + x_pts[-1]) / 2.0, max((x_pts[-1] - x_pts[0]) / 2.0, 1e-3)
        cz, rz = (z_pts[0] + z_pts[-1]) / 2.0, max((z_pts[-1] - z_pts[0]) / 2.0, 1e-3)
        
        dist = ((x_mesh - cx) / rx) ** 2 + ((z_mesh - cz) / rz) ** 2
        grid[:, y, :] = (dist <= 1.0).astype(np.float32)
        
    return gaussian_filter(grid, sigma=1.3)

def sample_blue_noise_surface(grid, num_points=78000):
    """Extracts marching cubes surface and applies Poisson-disk blue noise decimation."""
    verts, faces, normals, _ = measure.marching_cubes(grid, level=0.45)
    
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    prob = areas / np.sum(areas)
    
    sample_count = num_points * 2
    f_idx = np.random.choice(len(faces), size=sample_count, p=prob)
    
    r1 = np.sqrt(np.random.rand(sample_count, 1))
    r2 = np.random.rand(sample_count, 1)
    u, v, w = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
    
    pts = u * verts[faces[f_idx, 0]] + v * verts[faces[f_idx, 1]] + w * verts[faces[f_idx, 2]]
    nrms = u * normals[faces[f_idx, 0]] + v * normals[faces[f_idx, 1]] + w * normals[faces[f_idx, 2]]
    nrms /= np.linalg.norm(nrms, axis=1, keepdims=True) + 1e-8
    
    # Decimate to uniform spacing
    indices = np.random.choice(len(pts), size=num_points, replace=False)
    pts, nrms = pts[indices], nrms[indices]
    
    # Center and normalize coordinates to [-1, 1]
    center = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    pts = (pts - center) / (pts.max(axis=0) - pts.min(axis=0)).max()
    
    # Featureless face oval smoothing & dimming upward ledges
    up_dot = np.maximum(0.0, nrms[:, 1])
    bright = 0.60 + 0.40 * (1.0 - up_dot)
    
    # Smooth facial area
    face_sel = (pts[:, 1] > 0.05) & (pts[:, 1] < 0.42) & (pts[:, 2] > 0.08) & (np.abs(pts[:, 0]) < 0.22)
    bright[face_sel] *= 0.88
    
    rnd = np.random.rand(num_points).astype(np.float32)
    return pts.astype(np.float32), nrms.astype(np.float32), rnd, bright.astype(np.float32)

def export_binary(pts, nrms, rnd, bright, filename="dots.bin"):
    """Encodes point data in a compact 32-byte binary format."""
    n = len(pts)
    with open(filename, "wb") as f:
        f.write(struct.pack("<I", n))
        for i in range(n):
            f.write(struct.pack("<ffffffff",
                pts[i, 0], pts[i, 1], pts[i, 2],
                nrms[i, 0], nrms[i, 1], nrms[i, 2],
                rnd[i], bright[i]
            ))
    print(f"Baked {n} points into {filename} ({os.path.getsize(filename)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    cf_img = cv2.imread("clay-front.png", cv2.IMREAD_GRAYSCALE)
    cs_img = cv2.imread("clay-side.png", cv2.IMREAD_GRAYSCALE)
    photo = cv2.imread("photo.png")
    
    if cf_img is not None and cs_img is not None:
        cf_resized = cv2.resize(cf_img, (1200, 896))
        cs_resized = cv2.resize(cs_img, (1200, 896))
        
        mf = process_clay_mask(cf_resized)
        ms = process_clay_mask(cs_resized)
        
        vox = carve_voxel_hull(mf, ms, pitch=2.5)
        pts, nrms, rnd, br = sample_blue_noise_surface(vox, num_points=78000)
        export_binary(pts, nrms, rnd, br, "dots.bin")
    
    if photo is not None:
        cutout = cut_out_portrait(photo)
        cv2.imwrite("photo_cutout.png", cutout)
        print("Generated photo_cutout.png with soft alpha matting")
