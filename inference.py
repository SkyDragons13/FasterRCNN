import torch
import io
import sys
import psycopg2

from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from torchvision.ops import nms
from models import SwinBackbone,FasterRcnn
from PIL import Image
from psycopg2 import sql

PATH = "C:/##FACULTATE/Anul 3 Semestru 1/Practica/faster_rcnn.pth"
NAME = "licenta"
USER = "postgres"
PWD = "1q2w3e"
HOST = "localhost"
PORT = "5432"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

swin_fpn_backbone = SwinBackbone()
model = FasterRcnn(backbone=swin_fpn_backbone, num_classes=91,device=device)
model.load_state_dict(torch.load(PATH, weights_only=True,map_location=torch.device('cpu')))
model = model.to(device)
model.eval()

def resize_pil_image(image, min_size=600, max_size=1000):
    """
    Resize a PIL image to ensure the smaller dimension is at least min_size
    and the larger dimension is at most max_size, maintaining the aspect ratio.

    Args:
        image (PIL.Image.Image): Input image.
        min_size (int): Minimum size for the smaller dimension.
        max_size (int): Maximum size for the larger dimension.

    Returns:
        PIL.Image.Image: Resized image.
    """
    # Get original dimensions
    width, height = image.size
    min_original = min(width, height)
    max_original = max(width, height)

    # Compute scaling factor
    scale = min(min_size / min_original, max_size / max_original)

    # Compute new dimensions
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    # Ensure dimensions are clamped within the min and max size constraints
    new_width = max(min_size, min(new_width, max_size))
    new_height = max(min_size, min(new_height, max_size))

    # Resize image using bilinear resampling
    resized_image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    return resized_image

def get_image_from_db(image_id):
    conn = psycopg2.connect(dbname=NAME, user=USER, password=PWD, host=HOST, port=PORT)
    cursor = conn.cursor()
    cursor.execute("SELECT image FROM images WHERE id = %s", (image_id,))
    image_bytes = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def fetch_annotations_dict():
    """
    Fetches all annotations from the database and returns a dictionary
    mapping category_id to category_name.
    
    Returns:
        dict: A dictionary {category_id: category_name}
    """
    conn = psycopg2.connect(dbname=NAME, user=USER, password=PWD, host=HOST, port=PORT)
    cursor = conn.cursor()

    cursor.execute("SELECT category_id, category_name FROM annotations")
    rows = cursor.fetchall()

    annotations_dict = {category_id: category_name for category_id, category_name in rows}

    cursor.close()
    conn.close()

    return annotations_dict


def run_inference(image_id):
    # Load and preprocess the image
    image = get_image_from_db(image_id)
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Get the bounding boxes, labels, and scores
    pred_boxes = outputs['boxes'].squeeze(0).cpu()
    pred_labels = outputs['labels'].squeeze(0).cpu()
    pred_scores = outputs['scores'].squeeze(0).cpu()

    threshold = 0.5
    selected_indices = pred_scores >= threshold
    pred_boxes = pred_boxes[selected_indices]
    pred_labels = pred_labels[selected_indices]
    pred_scores = pred_scores[selected_indices]

    nms_indices = nms(pred_boxes, pred_scores, iou_threshold=0.3)
    pred_boxes = pred_boxes[nms_indices].numpy()
    pred_labels = pred_labels[nms_indices].numpy()
    pred_scores = pred_scores[nms_indices].numpy()

    image=resize_pil_image(image)
    draw=ImageDraw.Draw(image)
    font = ImageFont.load_default()

    cat_id_to_name=fetch_annotations_dict()

    pred_label_names = [cat_id_to_name[label] for label in pred_labels]

    for box, label_name, score in zip(pred_boxes, pred_label_names, pred_scores):
        box = list(map(int, box))  # Convert to integers
        draw.rectangle(box, outline="red", width=2)

        text = f'{label_name} {score:.2f}'
        # Calculate text size
        text_size = draw.textbbox((0, 0), text, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]

        # Text background rectangle (white)
        text_x = box[0]
        text_y = box[1] - text_height if box[1] - text_height > 0 else box[1]
        draw.rectangle(
            [(text_x, text_y), (text_x + text_width, text_y + text_height)],
            fill="white"
        )

        draw.text((text_x, text_y), f'{label_name} {score:.2f}', fill="black", font=font)

    # Convert image to bytes
    image_bytes_io = io.BytesIO()
    image.save(image_bytes_io, format='JPEG')
    image_bytes = image_bytes_io.getvalue()

    # Save to annotated_images table
    conn = psycopg2.connect(dbname=NAME, user=USER, password=PWD, host=HOST, port=PORT)

    cursor = conn.cursor()

    insert_query = sql.SQL("""
        INSERT INTO annotated_images (name, image, original_image_id)
        VALUES (%s, %s, %s)
    """)
    image_name = f"Ann{image_id}"
    cursor.execute(insert_query, (image_name, psycopg2.Binary(image_bytes), image_id))

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    image_id = sys.argv[1]

    # Run the inference on the image
    result = run_inference(image_id)

    print("Inference completed and saved to the database.")
