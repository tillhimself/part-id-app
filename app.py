import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# --- CONFIGURATION ---
DRAWINGS_DIR = "/Users/tillhead/shop_app/drawings"
st.set_page_config(page_title="Sheet Metal ID", page_icon="🛠️")

def identify_part(captured_image):
    """Matches the captured photo against the drawings using Feature Matching."""
    # Convert PIL image to OpenCV format
    file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
    img_query = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img_query, None)
    
    if des1 is None:
        return None, 0

    best_match_name = None
    max_matches = 0
    
    # Create Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Loop through drawings in the folder
    for filename in os.listdir(DRAWINGS_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(DRAWINGS_DIR, filename)
            img_train = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img_train is None:
                continue
                
            # Find features in drawing
            kp2, des2 = orb.detectAndCompute(img_train, None)
            
            if des2 is not None:
                # Match features
                matches = bf.match(des1, des2)
                # Filter for "good" matches
                good_matches = [m for m in matches if m.distance < 50]
                
                if len(good_matches) > max_matches:
                    max_matches = len(good_matches)
                    best_match_name = filename

    # Simple confidence calculation based on feature count
    confidence = min(100, (max_matches / 20) * 100) 
    return best_match_name, confidence

# --- USER INTERFACE ---
st.title("🛠️ Shop Part Identifier")
st.info(f"Scanning drawings in: `{DRAWINGS_DIR}`")

# Camera Input
camera_image = st.camera_input("Scan sheet metal part")

if camera_image:
    with st.spinner('Analyzing geometry...'):
        match_file, score = identify_part(camera_image)
        
        if match_file and score > 15: # Threshold to avoid false positives
            part_number = os.path.splitext(match_file)[0]
            
            st.success(f"### Match Found: {part_number}")
            st.metric("Confidence Score", f"{int(score)}%")
            
            # Show the reference drawing for verification
            ref_image_path = os.path.join(DRAWINGS_DIR, match_file)
            st.image(ref_image_path, caption=f"Reference Drawing: {match_file}", width=300)
        else:
            st.error("No match found. Try a clearer photo on a high-contrast background.")

# Sidebar Instructions
with st.sidebar:
    st.header("Instructions")
    st.write("1. Place part on a solid, dark background.")
    st.write("2. Ensure camera is directly above the part.")
    st.write("3. Avoid heavy shadows or glare on the metal.")
