# CS406_lab2
# Introduction
Ở trong bài lab này ta sẽ dùng đặc trưng histogram trong không gian màu HSV để tìm ra các ảnh gần giống với ảnh gốc nhất trong dataset có sẵn.  
Ứng dụng được sử dụng trên streamlit các giá trị hist được tính toán trước và lưu vào trong file data.json khi clone code về hãy giải nén file đó ra để sử dụng.  
File `preprocess.py` dùng để xử lý và tính toán hist trong dataset và lưu vào file data.json (Chúng ta không cần quan tâm lắm đến file này vì data đã được chạy sẵn).  
File `web.py` dùng để xây dựng giao diện và vận hành web
# Quickstart
## File structure
Lưu ý: phải lưu file như bên dưới để web chạy nếu không hãy sửa các đường dẫn trong file `web.py`
File  
├── dataset  
│   ├── seg  
│   └── seg_test  
├── preprocess.py  
├── web.py  
└── data.json  

## Download dataset
Link : <a href="https://drive.google.com/file/d/1F6sPtl0H-Sh7XPrAojDKcz_rBoUl_fgu/view?usp=sharing">Dataset Lab2</a>
## Install requirements
```bash
 pip install -r requirements.txt
```
## Calculate histogram
Chạy file `preprocess.py` để tính với data mới
```bash
 python preprocess.py
```
## Run web
```bash
python -m streamlit run web.py
```
## Demo


https://github.com/user-attachments/assets/fdc89d64-5ccc-4167-9d47-db5f35e1d512

