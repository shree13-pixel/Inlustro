
from extract_features import extract_all_features, extract_feature
from match_features import build_faiss_index, find_matches
import sys

# Paths
logo_folder = 'data/logos'
test_image = sys.argv[1]

# Step 1: Extract features
logo_features, logo_paths = extract_all_features(logo_folder)
test_feature = extract_feature(test_image)

# Step 2: Build FAISS index
index = build_faiss_index(logo_features)

# Step 3: Match test image with known logos
match_path, distance = find_matches(index, test_feature, logo_paths)

if match_path:
    print(f"[!] Potential IP Violation Detected: Similar to {match_path} (Distance: {distance:.4f})")
else:
    print("[✓] No IP violation detected.")
