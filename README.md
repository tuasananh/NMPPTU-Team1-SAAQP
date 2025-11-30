# Self-adaptive algorithms for quasiconvex programming and applications to machine learning

**Tài liệu gốc:** [Self-adaptive algorithms for quasiconvex programming and applications to machine learning (2024)](paper.pdf).

**Mục tiêu:** Đọc hiểu, Cài đặt thuật toán, Viết báo cáo (LaTex) và Thuyết trình (Beamer).

## Yêu cầu Chung

- **Ngôn ngữ:** Báo cáo và Slide viết bằng Tiếng Việt. Code comment bằng tiếng Anh.
- **Công cụ:**
  - Code: Python 3.8+, PyTorch, NumPy, Matplotlib.
  - Báo cáo: LaTeX (nên dùng local vì nhiều người).
  - Slide: LaTeX Beamer.
- **Deadline:** 30/12

## Chi tiết yêu cầu kết quả từng Team (mỗi team tự tạo plan để hoàn thành)

### Team 1: Báo cáo

- **Thư mục:** `report/`

- **Sản phẩm:** File PDF Báo cáo (LaTeX).

- **Yêu cầu sản phẩm:**
    - [ ] Dịch và định nghĩa chính xác các khái niệm toán học liên quan trong paper
    - [ ] Giải thích và chứng minh tính chính xác của ba thuật toán được đưa ra trong paper: GDA, DA và SGDA.
    - [ ] Khi có kết quả của team 2, thêm kết quả chạy các thuật toán (biểu đồ, v.v) vào báo cáo.
    - [ ] File PDF báo cáo bằng tiếng Việt, không lỗi font, đầy đủ Mục lục và Tài liệu tham khảo (References).
    - [ ] Các file .tex liên quan để trong thư mục `report/src`, file pdf kết quả đặt ở `report/report.pdf`

### Team 2: Cài đặt thuật toán 

- **Thư mục:** `implementation/`

- **Sản phẩm:** Các thuật toán và các ví dụ ứng dụng thuật toán.

- **Yêu cầu sản phẩm:**:
    - [ ] 3 thuật toán chính cài đặt để ứng dụng vào bất kỳ bài toán nào như black box
    - [ ] Giải ví dụ 1 trong paper (Page 8) và kết quả so sánh các thuật toán
    - [ ] Giải ví dụ 2 trong paper (Page 9) và kết quả so sánh các thuật toán
    - [ ] Giải ví dụ 3 trong paper (Page 10) và kết quả so sánh các thuật toán
    - [ ] Giải ví dụ Multivariable Logistic Regression (Page 13) 
    - [ ] Giải ví dụ Neural Network (Page 13)
    - [ ] Source code được lưu ở `implementation/src/` và các kết quả được lưu ở `implementation/output/`


### Team 3: Slides beamer

- **Thư mục:** `slides/`

- **Sản phẩm:** File PDF Slide (Beamer).

- **Yêu cầu sản phẩm:**
    - [ ] Định nghĩa chính xác các khái niệm toán học liên quan trong paper (vắn tắt)
    - [ ] Giải thích ba thuật toán được đưa ra bao gồm pseudocode trong paper: GDA, DA và SGDA và chứng minh tính chính xác của chúng.
    - [ ] Khi có kết quả của team 2, thêm kết quả chạy các thuật toán (biểu đồ, v.v) vào báo cáo.
    - [ ] Slides trình bày rõ ràng và mạch lạc
    - [ ] Các file .tex liên quan để trong thư mục `slides/src`, file pdf kết quả đặt ở `slides/slides.pdf`

## Cấu trúc thư mục repo tham khảo

```
NMPPTU-Team1-SAAQP/
├── .gitignore                  <-- File cấu hình chặn file rác/data nặng lên git
├── README.md                   <-- Hướng dẫn chung
├── requirements.txt            <-- Liệt kê thư viện Python (numpy, torch, matplotlib...)
├── data/                       <-- Nơi chứa Datasets (W8a, Mushrooms, Cifar10)
│
├── report/                     <-- 🟡 KHÔNG GIAN LÀM VIỆC TEAM 1
│   ├── report.pdf              <-- [Sản phẩm cuối] File báo cáo hoàn chỉnh
│   └── src/                    <-- Source code LaTeX
│       ├── main.tex            <-- File chính gọi các chương
│       ├── references.bib      <-- File trích dẫn tài liệu tham khảo
│       ├── chapters/           <-- Chia nhỏ nội dung để nhiều người cùng viết
│       │   ├── 1_introduction.tex
│       │   ├── 2_preliminaries.tex
│       │   └── 3_proofs.tex
│       └── images/             <-- Chứa ảnh biểu đồ (Copy từ implementation/output qua)
│
├── implementation/             <-- 🟢 KHÔNG GIAN LÀM VIỆC TEAM 2
│   ├── output/                 <-- [Sản phẩm cuối] Nơi code xuất ra kết quả
│   │   ├── figures/            <-- Ảnh biểu đồ (Figure 1, 2, 6, 7...)
│   │   └── logs/               <-- File text kết quả so sánh (Table 1, 2)
│   │
│   └── src/                    <-- Source code Python
│       ├── algorithms/         <-- Code 3 thuật toán "Black box" (Core)
│       │   ├── __init__.py
│       │   ├── gda.py          <-- Alg 1: Gradient Descent Adaptive
│       │   ├── gd.py           <-- Alg 2: Gradient Descent (Cổ điển)
│       │   └── sgda.py         <-- Alg 3: Stochastic GDA (Deep Learning)
│       │
│       └── examples/           <-- Code giải các ví dụ cụ thể (Gọi module algorithms)
│           ├── ex1_nonconvex.py
│           ├── ex2_nonsmooth.py
│           ├── ex3_comparison.py
│           ├── ml_logistic.py
│           └── ml_resnet_network.py
│
└── slides/                     <-- 🔵 KHÔNG GIAN LÀM VIỆC TEAM 3
    ├── slides.pdf              <-- [Sản phẩm cuối] File trình chiếu
    └── src/                    <-- Source code Beamer
        ├── main.tex            <-- File chính
        ├── sections/           <-- Các phần nội dung slide
        │   ├── theory.tex
        │   └── experiments.tex
        └── media/              <-- Ảnh biểu đồ dùng cho slide
```