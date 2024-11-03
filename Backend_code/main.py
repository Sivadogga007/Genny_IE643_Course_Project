from fastapi import FastAPI, File, UploadFile,BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pdf2image import convert_from_path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import tempfile
from pathlib import Path
import subprocess
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import cv2
from CRAFT import CRAFTModel, draw_polygons
app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VisionEncoderDecoderModel.from_pretrained('checkpoint-13540').to(device)
polygon_model = CRAFTModel('weights', 'cpu', use_refiner=True, fp16=True)

PERMANENT_DIR = "pdf_images"
Path(PERMANENT_DIR).mkdir(parents=True, exist_ok=True)  # Creates the directory if it doesn't exist
import os

def load_image(image_path):
    """Load an image from a file."""
    return cv2.imread(image_path)

def crop_polygons(image, polygons):
    """
    Crop images based on the provided polygons.
    :param image: The original image.
    :param polygons: List of polygons defining the areas to crop.
    :return: List of cropped images and their y-positions.
    """
    cropped_images = []
    y_positions = []
    
    for polygon in polygons:
        # Create a mask for the polygon
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)

        # Get the bounding box of the polygon
        x, y, w, h = cv2.boundingRect(np.array(polygon, dtype=np.int32))
        
        # Crop the image using the mask
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        cropped_image = masked_image[y:y+h, x:x+w]

        # Resize the cropped image for the text recognition model
        cropped_image = cv2.resize(cropped_image, (640, 480))
        
        cropped_images.append(cropped_image)
        y_positions.append(np.mean([point[1] for point in polygon]))  # Mean y-coordinate

    return cropped_images, y_positions

def recognize_text_from_images(cropped_images):
    """Recognize text from the cropped images using the model."""
    recognized_texts = []
    for cropped_image in cropped_images:
        image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        recognized_texts.append(generated_text)
    return recognized_texts


@app.post("/upload_pdf/")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return JSONResponse(content={"message": "Invalid file type"}, status_code=400)

    # Save the uploaded PDF file to the current working directory
    pdf_path = os.path.join(os.getcwd(), file.filename)
    with open(pdf_path, "wb") as pdf_file:
        pdf_file.write(await file.read())

    # Convert PDF pages to images
    images = convert_from_path(pdf_path, dpi=200, fmt="jpeg")

    # Save images to the current working directory and collect their paths
    image_paths = []
    for i, image in enumerate(images):
        
        # img = Image.open('/kaggle/input/nadikcew/eng_AF_004.jpg')
        polygons = polygon_model.get_polygons(image)
        # result = draw_polygons(images, polygons)
        image_path = os.path.join(os.getcwd(), f"{os.path.splitext(file.filename)[0]}_page_{i + 1}.jpeg")
        image.save(image_path, "JPEG")
        image_paths.append(image_path)
            # Crop images and retrieve their y-positions
        cropped_images, y_positions = crop_polygons(image, polygons)

        # Recognize text from cropped images
        recognized_texts = recognize_text_from_images(cropped_images)

        # Combine recognized texts into paragraphs based on y-positions
        merged_paragraphs = []
        current_paragraph = ""
        threshold = 70  # Define a threshold for merging based on y-coordinate proximity

        for i, text in enumerate(recognized_texts):
            if text.strip():  # If the recognized text is not empty
                if current_paragraph:  # If there is already some text in the paragraph
                    # Check if current and last y-position are within the threshold
                    if abs(y_positions[i] - y_positions[i - 1]) < threshold:
                        current_paragraph += " " + text.strip()  # Merge

                    else:
                        merged_paragraphs.append(current_paragraph.strip())
                        current_paragraph = text.strip()  # Start a new paragraph
                else:
                    current_paragraph = text.strip()  # Start a new paragraph

        # Append any remaining text
        if current_paragraph:
            merged_paragraphs.append(current_paragraph.strip())

        # Print the recognized text paragraphs
        print("Recognized Text Paragraphs:")
        for i, paragraph in enumerate(merged_paragraphs):
            text+=f"""
\\textbf{{Paragraph {i+1}}}: \textcolor{{blue}}{{\\underline{{{paragraph}}}}} \\\\
\\vspace{{10pt}} % Space after each paragraph
"""
        text+= r""" 
        \\newpage
        """        

    # Minimal LaTeX code for testing
        # Minimal LaTeX code for testing
    header = r"""
   \documentclass[preprint,authoryear]{elsarticle}
\usepackage{amssymb}
\journal{}%Mathematical Biosciences}
\usepackage{graphicx}
\usepackage{amsmath,amssymb,amsfonts,latexsym}
\usepackage{color}
\usepackage{eucal}%    caligraphic-euler fonts: \mathcal{ }
\usepackage{xspace}
\usepackage{hyperref}
\usepackage{bm}
\usepackage[title]{appendix}
\newtheorem{proposition}{Proposition}
\newtheorem{proof}{Proof}
\usepackage[round]{natbib}
\usepackage{algorithm}
\usepackage[noend]{algpseudocode}
\DeclareMathOperator*{\argmax}{argmax} % no space, limits underneath in displays
\DeclareMathOperator*{\argmin}{argmin} % no space, limits underneath in displays
\begin{document}
    """
    footer=r"""
    \end{document}"""
    
    latex_code = header+text+footer
    # Write LaTeX code to file in the current working directory
    latex_file_path = os.path.join(os.getcwd(), "test_output.tex")
    with open(latex_file_path, "w") as f:
        f.write(latex_code)

    # Compile the LaTeX file
    process = subprocess.run(['pdflatex', latex_file_path], capture_output=True, text=True)

    if process.returncode != 0:
        print("LaTeX compilation failed:")
        print(process.stderr)
        return JSONResponse(content={"message": "LaTeX compilation failed"}, status_code=500)
    else:
        pdf_output_path = os.path.join(os.getcwd(), "test_output.pdf")

        # Add cleanup tasks to remove files after response
        background_tasks.add_task(cleanup_files, [pdf_path, latex_file_path, pdf_output_path] + image_paths)

        # Return the compiled PDF as a response
        return FileResponse(pdf_output_path, media_type='application/pdf', filename="test_output.pdf")

def cleanup_files(file_paths):
    """Delete files from provided paths."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)