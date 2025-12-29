# CleanShot - Image Text Removal Tool

## Overview

CleanShot is a web-based image processing application that removes text and numbers from images using OCR (Optical Character Recognition) and image inpainting techniques. Users can upload images through a drag-and-drop interface, and the application processes them to remove detected text while preserving the underlying image content.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Framework**: Flask (Python web framework)
- **Image Processing Pipeline**:
  1. Images are uploaded and stored in `static/uploads/`
  2. OpenCV reads and processes images
  3. Tesseract OCR detects text regions with confidence scores
  4. A mask is generated for text areas (with slight padding for edge coverage)
  5. OpenCV's TELEA inpainting algorithm fills in masked regions
  6. Processed images are saved to `static/processed/`

### Frontend Architecture
- **Template Engine**: Jinja2 (Flask's default)
- **Styling**: Tailwind CSS via CDN
- **Features**: 
  - Drag-and-drop file upload zone
  - Loading spinner during processing
  - Progressive Web App (PWA) support

### PWA Implementation
- Service worker (`sw.js`) provides offline caching
- Web manifest enables "Add to Home Screen" functionality
- Caches static assets and CDN resources

### File Handling
- Maximum upload size: 16MB
- Supported formats: JPG, PNG, WEBP
- Files are secured using Werkzeug's `secure_filename`

## External Dependencies

### Python Libraries
- **Flask**: Web framework for routing and request handling
- **OpenCV (cv2)**: Image processing and inpainting
- **pytesseract**: Python wrapper for Tesseract OCR engine
- **NumPy**: Array operations for image masks
- **Pillow (PIL)**: Additional image handling support

### System Dependencies
- **Tesseract OCR**: Must be installed on the system for text detection to work

### CDN Resources
- Tailwind CSS for styling
- Flaticon for PWA icon assets

### Storage
- Local filesystem storage in `static/uploads/` and `static/processed/` directories
- No database required - stateless image processing