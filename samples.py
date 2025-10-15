
import os
import tempfile
import json
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models import SwinBackbone,FasterRcnn,FilteredCocoDetection
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw, ImageFont

def sample(small_loader,device,cat_id_to_name,model):
    """
    Runs inference on a small data loader and visualizes the predicted bounding boxes
    with class labels and confidence scores on the images.

    This function:
    - Moves images to the given device.
    - Performs inference using the given model.
    - Converts the image and bounding box results into a visual format.
    - Displays the image with drawn bounding boxes, category names, and scores.

    Args:
        small_loader (torch.utils.data.DataLoader): A DataLoader yielding a small batch of samples.
            Assumes each item is a tuple of (image, target), where `image` is a tensor and
            `target` is a dict with metadata.
        device (torch.device): The device on which the model should run.
        cat_id_to_name (dict): A dictionary mapping category IDs to category names.
        model (torch.nn.Module): The trained object detection model.

    Notes:
        - This function assumes the model returns a dictionary with keys: 'boxes', 'labels', and 'scores'.
        - It uses a confidence threshold of 0.5 to filter predictions.
        - The resized image maintains aspect ratio within bounds using `resize_pil_image`.
    """
    model.eval()
    for i, (images, targets) in enumerate(small_loader):
        images = images.to(device)

        if i == 10 :
            break # Stops the loop at 10 images , this function is only for samples from the validation set

        # Run inference
        with torch.no_grad():
            outputs = model(images)

        pred_boxes = outputs['boxes'].cpu().numpy()
        pred_labels = outputs['labels'].cpu().numpy()
        pred_scores = outputs['scores'].cpu().numpy()

        # Convert image tensor to a PIL image
        image_pil = transforms.ToPILImage()(images[0].cpu())
        image_pil=resize_pil_image(image_pil)

        # Visualization
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.load_default()

        # Filter predictions with a threshold (e.g., 0.5)
        threshold = 0.5
        selected_indices = pred_scores >= threshold
        pred_boxes = pred_boxes[selected_indices]
        pred_labels = pred_labels[selected_indices]
        pred_scores = pred_scores[selected_indices]

        # Map category IDs to names
        pred_label_names = [cat_id_to_name[label] for label in pred_labels]

        # Draw boxes and labels
        for box, label_name, score in zip(pred_boxes, pred_label_names, pred_scores):
            box = list(map(int, box))  # Convert to integers
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]), f'{label_name} {score:.2f}', fill="black", font=font)
            print(box)

        # Display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(image_pil)
        plt.axis('off')
        plt.title(f"Image {i+1}")
        plt.show()

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

def evaluate_coco(model, val_loader, device):
    """
    Evaluates a Faster R-CNN (or similar) object detection model on a validation dataset
    using COCO metrics and returns the mean Average Precision (mAP).

    This function:
    - Converts model outputs to COCO-format results.
    - Saves results temporarily in JSON format.
    - Uses COCO's official evaluation tools (pycocotools) to compute metrics.

    Args:
        model (torch.nn.Module): The trained detection model.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
            Assumes the dataset is based on `CocoDetection`.
        device (torch.device): Device on which to run evaluation.

    Returns:
        float: mAP (mean Average Precision) @ IoU=0.50:0.95

    Notes:
        - This process will take at least 30 mins to complete 
    """
    model.eval()
    base_dataset = val_loader.dataset
    if isinstance(base_dataset, torch.utils.data.Subset):
        base_dataset = base_dataset.dataset

    coco_gt = base_dataset.coco_api    
    coco_results = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)  # Output: list of dicts with 'boxes', 'scores', 'labels'

            for i, output in enumerate([outputs]):
                image_id = targets[i]['image_id'].item()
                boxes = output['boxes'].cpu()
                scores = output['scores'].cpu()
                labels = output['labels'].cpu()

                boxes_xywh = torchvision.ops.box_convert(boxes, in_fmt='xyxy', out_fmt='xywh')

                for box, score, label in zip(boxes_xywh, scores, labels):
                    coco_results.append({
                        'image_id': image_id,
                        'category_id': int(label),
                        'bbox': box.tolist(),
                        'score': float(score)
                    })

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        json.dump(coco_results, f)
        result_file = f.name

    coco_dt = coco_gt.loadRes(result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]  # mAP @ IoU=0.50:0.95


# For Tensorboard run <tensorboard --logdir=./logs --port=6006> in the terminal (without the <>) and open http://localhost:6006 in your browser

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the full dataset from the downloaded COCO dataset for training
    dataset = FilteredCocoDetection(root='C:/##FACULTATE/Anul 3 Semestru 1/Practica/content/train2017', annFile='C:/##FACULTATE/Anul 3 Semestru 1/Practica/content/annotations/instances_train2017.json', transform=transforms.ToTensor())

    small_val_ds=torch.utils.data.Subset(dataset,range(10001,11001)) # We pick the next 1000 images for validating since the first 10000 are used for training 

    # Create data loaders using the filtered datasets (only images with targets)
    small_val_loader=DataLoader(small_val_ds,shuffle=False)

    # Wrap Swin-T Backbone
    swin_fpn_backbone = SwinBackbone()
    # Load COCO dataset metadata
    annotation_path = "C:/##FACULTATE/Anul 3 Semestru 1/Practica/content/annotations/instances_train2017.json"
    coco = COCO(annotation_path)

    # Create a dictionary mapping category IDs to names
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

    # Use in Faster R-CNN
    model = FasterRcnn(swin_fpn_backbone, num_classes=91,device=device)
    model = model.to(device)
    model.load_state_dict(torch.load('C:/##FACULTATE/Anul 3 Semestru 1/Practica/faster_rcnn.pth', weights_only=True,map_location=torch.device('cpu')))

    sample(small_val_loader,device,cat_id_to_name,model=model)

    log_dir="./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Run evaluation
    map_score = evaluate_coco(model, small_val_loader, device)
    print(f"Final mAP: {map_score:.4f}")

    # Log to TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_scalar('Metrics/Final_mAP', map_score)
    writer.close()

if __name__ == "__main__":
   main()
  