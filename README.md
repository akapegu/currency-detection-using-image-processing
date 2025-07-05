# Currency Detection Using Image Processing

A computer vision system that detects and classifies Indian Rupee and Japanese Yen banknotes using image processing techniques.

## Features

- **Automatic Currency Detection**: Identifies Indian Rupee (INR) or Japanese Yen (JPY)
- **Value Classification**: Determines specific denominations
- **Visual Output**: Shows original image with detected currency value
- **Processing Visualization**: Displays intermediate processing steps

## Supported Currencies

**Indian Rupee**: ₹10, ₹50, ₹100, ₹200, ₹500, ₹2000
**Japanese Yen**: ¥1000, ¥2000, ¥5000, ¥10000

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your currency image in the `images/` folder
2. Update the image path in `code.py` (line 235):
   ```python
   image = cv2.imread("images/your_image.jfif")
   ```
3. Run the script:
   ```bash
   python code.py
   ```

## How It Works

1. **Preprocessing**: Resize image to standard dimensions
2. **Edge Detection**: Apply custom Canny edge detection
3. **Currency Classification**: Analyze specific regions to determine currency type
4. **Value Detection**: Extract features and match to pre-defined datasets
5. **Output**: Display results in two windows

## Project Structure

```
├── code.py          # Main detection algorithm
├── README.md        # This file
├── requirements.txt # Dependencies
└── images/          # Currency images
    ├── *.jfif      # Test images
    └── *.jpg       # Display images
```

## Requirements

- Python 3.x
- OpenCV
- NumPy 