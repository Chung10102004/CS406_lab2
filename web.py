import streamlit as st
from PIL import Image
import cv2
import numpy as np
import json
import os
from numpy.linalg import norm
from scipy.spatial.distance import cosine, correlation, minkowski, cityblock

def load_histogram_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    histogram_data = {path: np.array(hist) for path, hist in data.items()}
    return histogram_data

def calc_hist(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def euclidean_distance(hist1, hist2):
    return norm(hist1 - hist2)

def cosine_distance(hist1, hist2):
    return cosine(hist1, hist2)

def correlation_distance(hist1, hist2):
    return correlation(hist1, hist2)

def manhattan_distance(hist1, hist2):
    return cityblock(hist1, hist2)

def minkowski_vectorized(histograms, input_hist, p):
    return np.sum(np.abs(histograms - input_hist) ** p, axis=1) ** (1/p)

def find_top_k_similar_vectorized(input_hist, dataset_histograms, metric='euclidean', k=10, p=3):
    paths = list(dataset_histograms.keys())
    histograms = np.array(list(dataset_histograms.values()))
    
    if metric == 'euclidean':
        distances = norm(histograms - input_hist, axis=1)

    elif metric == 'cosine':
        distances = np.array([cosine_distance(input_hist, hist) for hist in histograms])

    elif metric == 'correlation':
        distances = np.array([correlation_distance(input_hist, hist) for hist in histograms])
        
    elif metric == 'minkowski':
        distances = minkowski_vectorized(histograms, input_hist, p)

    elif metric == 'manhattan':
        distances = np.array([manhattan_distance(input_hist, hist) for hist in histograms])
    else:
        st.error("Phương thức tính khoảng cách không hợp lệ.")
        return []
    
    top_k_indices = np.argsort(distances)[:k]
   
    top_k = [{'path': paths[i], 'distance': distances[i]} for i in top_k_indices]
    
    return top_k


def load_image(image_path):
    return Image.open(image_path)

def main():
    st.title("Ứng Dụng Tìm Kiếm Ảnh Tương Tự Bằng Histogram")
    
    st.sidebar.header("Cài Đặt Tìm Kiếm")
    distance_metric = st.sidebar.selectbox(
        "Chọn phương thức tính khoảng cách:",
        ("Euclidean", "Cosine", "Correlation", "Minkowski", "Manhattan")
    )
    
    if distance_metric == "Minkowski":
        p = st.sidebar.slider("Chọn giá trị p cho Minkowski Distance:", min_value=1, max_value=10, value=3, step=1)
    else:
        p = 3  
    
   
    metric_mapping = {
        "Euclidean": "euclidean",
        "Cosine": "cosine",
        "Correlation": "correlation",
        "Minkowski": "minkowski",
        "Manhattan": "manhattan"
    }
    selected_metric = metric_mapping.get(distance_metric, "euclidean")
    
    json_path = "data1.json"  # Đường dẫn tới file JSON chứa histogram
    if not os.path.exists(json_path):
        st.error(f"Không tìm thấy file histogram tại đường dẫn: {json_path}")
        return
    
    dataset_histograms = load_histogram_data(json_path)
    st.write(f"Đã tải {len(dataset_histograms)} histogram từ dataset.")
 
    st.header("Tải Ảnh Đầu Vào")
    uploaded_file = st.file_uploader("Chọn một ảnh để tìm kiếm", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Ảnh Đầu Vào', use_column_width=True)

        # Chuyển đổi ảnh từ PIL sang OpenCV định dạng BGR
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        input_hist = calc_hist(image_cv)
        
        with st.spinner('Đang tìm kiếm ảnh tương tự...'):
            top_k = find_top_k_similar_vectorized(
                input_hist, 
                dataset_histograms, 
                metric=selected_metric, 
                k=10, 
                p=p
            )
        
        st.success("Đã tìm kiếm xong!")
        
        if top_k:
            st.header("Top 10 Ảnh Tương Tự")
            num_cols = 5  
            cols = st.columns(num_cols)
            
            for idx, result in enumerate(top_k):
                image_path = result['path']
                distance = result['distance']
                label = os.path.basename(os.path.dirname(image_path))  
                if os.path.exists(image_path):
                    similar_image = load_image(image_path)
                    col = cols[idx % num_cols]
                    with col:
                        st.image(similar_image, use_column_width=True, caption=f"Label: {label}; Khoảng cách: {distance:.4f}")
                else:
                    st.warning(f"Không tìm thấy ảnh tại đường dẫn: {image_path}")

if __name__ == "__main__":
    main()
