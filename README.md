# 🧠 Image Enhancement Techniques for Poor Quality Images Using Classical Computer Vision

This repository contains a comprehensive image enhancement project focused on improving the visual quality of poor-quality images using classical image processing techniques. The project includes tasks like component extraction, blur correction, noise reduction, and complex visual enhancement. All operations are performed using Python and OpenCV.

---

## 📁 Repository Structure

```bash
📦 image-enhancement-poor-quality
├── Image Enhancement for Selected Poor Quality Images/
│   ├── [Contains: code notebooks, processed & original images]
├── Report.docx
└── README.md
```

> 📝 **Note:** All implementation code and image data are organized within the folder `Image Enhancement for Selected Poor Quality Images`.

---


## 🎯 Objective

The objective is to evaluate the effectiveness of classical image processing algorithms on a variety of poor-quality image types. The tasks involve applying enhancement techniques in the following categories:

1. **Component Extraction**  
2. **Blurred Image Enhancement**  
3. **Noise Removal**  
4. **Complex Visual Enhancement**

---

## 🧪 Techniques Employed

Each image category has been enhanced using one or more of the following techniques:

### Component Extraction

#### 🟠 Techniques:
- Grayscale Conversion
- Gaussian Blurring
- Laplacian Edge Detection
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Hough Circle Transform
- HSV Color Segmentation
- Adaptive Thresholding
- Contour Detection

#### 📌 Sample Code
```python
# Grayscale + CLAHE
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_clahe = clahe.apply(gray)
```

```python
# Circle Detection using Hough Transform
circles = cv2.HoughCircles(
    gray_clahe,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=50,
    param1=50,
    param2=30,
    minRadius=10,
    maxRadius=100
)
```

---

### Blurred Image Enhancement

#### 🟠 Techniques:
- Unsharp Masking
- Laplacian Sharpening
- Frequency Domain High-Pass Filter
- Sobel Edge Enhancement
- Custom Sharpening Kernel

#### 📌 Sample Code
```python
# Unsharp Masking
gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)
unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
```

```python
# Sobel Edge Detection
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobelx, sobely)
```

---

### Noise Removal

#### 🟠 Techniques:
- Median Filtering
- Gaussian Filtering
- Non-Local Means Denoising
- Adaptive Thresholding

#### 📌 Sample Code
```python
# Median Filtering
median_filtered = cv2.medianBlur(noisy_img, 5)

# Non-Local Means Denoising
denoised = cv2.fastNlMeansDenoisingColored(
    noisy_img, None, h=10, templateWindowSize=7, searchWindowSize=21
)
```

---

### Challenging Image Enhancement

#### 🟠 Techniques:
- Histogram Equalization
- CLAHE
- Sharpening with Custom Kernel
- Binarization (Thresholding)

#### 📌 Sample Code
```python
# Histogram Equalization
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
```

```python
# Binarization
_, binary = cv2.threshold(equalized, 127, 255, cv2.THRESH_BINARY)
```

---

## 📊 Visual Results

| Task                        | Technique                           | Result Description |
|-----------------------------|-------------------------------------|--------------------|
| Circle Detection            | CLAHE + Hough Circle Transform      | Clear, accurate circle marking |
| COVID-19 Chart              | HSV + Adaptive Threshold            | Segmented by color (blue, green, orange) |
| Building Sharpening         | Unsharp + Sobel                     | Sharper edges, clearer structure |
| Dog Image Sharpening        | Custom Kernel                       | Enhanced texture detail |
| Text Noise Removal          | Median + Non-Local Means            | Legible, low noise text |
| Rocket Image Denoising      | Non-Local Means                     | Clean image with preserved detail |
| Wind Chart Enhancement      | CLAHE + Adaptive Threshold          | Better contrast and readability |
| Newspaper Visual Clarity    | CLAHE + Binarization                | Text enhanced for OCR |
| Name Plate Visibility       | CLAHE + Sharpen + Binarization      | Enhanced contrast and edge definition |

---

## 📈 Discussion & Analysis

### ✅ Pros
- **Modular Pipeline:** Each task uses a specific method pipeline tailored to its image defect type.
- **Contrast Techniques:** CLAHE and Histogram Equalization significantly improved detail in poor lighting.
- **Noise Removal:** Non-Local Means Denoising retained most fine details, especially useful for text and charts.
- **Sharpening:** Unsharp masking and Sobel-based methods were most effective for natural images like buildings and animals.

### ❌ Cons
- Over-sharpening introduced halos and artifacts in some cases.
- Thresholding led to some detail loss in highly textured images.
- HSV segmentation can misfire when color overlap is present.
- Noise-removal methods like median filtering slightly blurred edges.

---

## 🔍 Suggestions for Future Work

- Use **Canny Edge Detection** before transformations like Hough to improve accuracy.
- Integrate **deep learning models** such as SRCNN for super-resolution tasks.
- Apply **K-means or Watershed segmentation** to isolate components in more complex scenes.
- Consider advanced denoising like **BM3D** or **Wavelet Shrinkage**.
- Add automatic parameter tuning using quality metrics like PSNR and SSIM.

---

## 🧰 Dependencies

```txt
Python >= 3.8
opencv-python
numpy
matplotlib
scikit-image
jupyter
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Clone the Repository
```bash
git clone https://github.com/your-username/image-enhancement-cv-mini-project.git
cd image-enhancement-cv-mini-project
```

### Run the Notebook
```bash
jupyter notebook DSAI_352_Mohamed_Ehab_Yousri.ipynb
```

---

## 📚 References

- Gonzalez, R. C., & Woods, R. E. *Digital Image Processing* (4th ed.)
- OpenCV Documentation: [https://docs.opencv.org/](https://docs.opencv.org/)
- Scikit-Image Docs: [https://scikit-image.org/docs/stable/](https://scikit-image.org/docs/stable/)
- PyImageSearch Tutorials: [https://pyimagesearch.com/](https://pyimagesearch.com/)

---

---
