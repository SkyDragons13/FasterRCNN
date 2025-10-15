import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor
from torchvision.datasets import CocoDetection
from torchvision.models.swin_transformer import swin_t,Swin_T_Weights
from torchvision.ops import MultiScaleRoIAlign,nms,box_convert,box_iou
from typing import List, Tuple

class FilteredCocoDetection(CocoDetection):
    """
    A subclass of `torchvision.datasets.CocoDetection` that filters out images without valid annotations used if you want to calculate mAp using cocoeval.

    This class ensures that the dataset only includes images with at least one annotation.
    This is particularly useful for training and evaluation, as COCO can include images
    with no labeled objects, which can lead to issues during model training or inference.

    Attributes:
        coco_api (pycocotools.coco.COCO): Reference to the COCO API object.
        ids (list): Filtered list of image IDs that have at least one annotation.

    Methods:
        __getitem__(index):
            Returns the image and annotations at the given index.
            Raises an IndexError if no annotations are found (which should not happen due to pre-filtering).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.coco_api=self.coco
        # Filter out image IDs that have no valid annotations
        self.ids = [img_id for img_id in self.ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0]

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        # Optional: further filter out corrupted or weird annotations
        if len(target) == 0:
            # shouldn't happen if self.ids was filtered, but just in case
            raise IndexError("No annotations found for this index.")

        return img, target

class ImageList:
    """
    Structure that holds a list of images (of possibly varying sizes) as a single tensor.
    This works by padding the images to the same size, and storing in a field the original sizes of each image (later done in the model)

    Args:
        tensors (tensor): Tensor containing images.
        image_sizes (list[tuple[int, int]]): List of Tuples each containing size of images.
    """

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]) -> None:
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> "ImageList":
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)

class RPN(nn.Module):
    def __init__(self, in_channels=512, anchors=9):
        """
        Region Proposal Network used in Faster R-CNN.

        Args:
            in_channels (int): Number of input channels from the backbone feature map.
            anchors (int): Number of anchor boxes per spatial location (usually scales * aspect_ratios).
        """
        super(RPN, self).__init__()

        # Shared 3x3 convolution for feature extraction
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # 1x1 convolution to predict objectness logits (foreground vs. background)
        self.cls_logits = nn.Conv2d(in_channels, anchors, kernel_size=1, stride=1)

        # 1x1 convolution to predict bounding box deltas for each anchor
        self.bbox_pred = nn.Conv2d(in_channels, anchors * 4, kernel_size=1, stride=1)

        # Initialize weights for all layers
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

        # Store number of anchors per spatial location
        self.anchors = anchors

    def forward(self, x):
        """
        Forward pass through the RPN.

        Args:
            x (list[Tensor]): List of feature maps from different Swin stages.
                              Each feature map has shape (N, C, H, W)

        Returns:
            logits (Tensor): Objectness scores of shape (N * H * W * anchors, 1)
            bbox_preds (Tensor): Predicted bbox deltas of shape (N * H * W * anchors, 4)
        """
        logits = []
        bbox_preds = []

        for feature in x:
            # Shared conv + ReLU
            t = F.relu(self.conv(feature))  # Shape: (N, C, H, W)

            # Compute objectness logits
            cls_score = self.cls_logits(t)                # (N, K, H, W)
            cls_score = cls_score.permute(0, 2, 3, 1)     # (N, H, W, K)
            cls_score = cls_score.reshape(-1, 1)          # (N*H*W*K, 1)
            logits.append(cls_score)

            # Compute bounding box regression outputs
            box_transform_pred = self.bbox_pred(t)        # (N, K*4, H, W)
            box_transform_pred = box_transform_pred.view(
                box_transform_pred.size(0),               # Batch size N
                self.anchors,                             # K anchors
                4,                                        # 4 box parameters
                t.shape[-2],                              # Height H
                t.shape[-1]                               # Width W
            )                                             # Shape: (N, K, 4, H, W)
            box_transform_pred = box_transform_pred.permute(0, 3, 4, 1, 2)  # (N, H, W, K, 4)
            box_transform_pred = box_transform_pred.reshape(-1, 4)          # (N*H*W*K, 4)
            bbox_preds.append(box_transform_pred)

        # Concatenate across all levels (if using multi-scale features)
        logits = torch.cat(logits, dim=0)
        bbox_preds = torch.cat(bbox_preds, dim=0)

        return logits, bbox_preds
    
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, roi_output_size):
        """
        Detection head for the Faster R-CNN model.
        It takes in features pooled from RoIs and predicts classification scores and bounding box adjustments.

        Args:
            in_channels (int): Number of input channels from the backbone (e.g., 256, 512).
            num_classes (int): Number of object classes to classify (including background).
            roi_output_size (int): The height/width of the RoI pooled feature map (e.g., 7 for 7x7).
        """
        super(DetectionHead, self).__init__()

        # Fully connected layers for processing flattened RoI features
        self.fc1 = nn.Linear(in_channels * roi_output_size ** 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        # Final classification layer: outputs logits for each class
        self.cls_score = nn.Linear(1024, num_classes)

        # Final regression layer: outputs bounding box deltas for each class
        # Output shape: (N, num_classes * 4) -> each box has [dx, dy, dw, dh]
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

        # Initialize weights: normal distribution for weights, 0 for biases
        for layer in [self.fc1, self.fc2, self.cls_score, self.bbox_pred]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Forward pass of the detection head.

        Args:
            x (Tensor): Input RoI-pooled features of shape (N, C, H, W).

        Returns:
            cls_logits (Tensor): Classification scores of shape (N, num_classes).
            bbox_deltas (Tensor): Bounding box regression outputs of shape (N, num_classes * 4).
        """
        # Flatten RoI features from (N, C, H, W) -> (N, C*H*W)
        x = x.flatten(start_dim=1)

        # Pass through two fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Predict class logits and bounding box deltas
        cls_logits = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return cls_logits, bbox_deltas

class SwinBackbone(nn.Module):
    """
    SwinBackbone wraps a pretrained Swin Transformer (Swin-T) and adapts its outputs
    for use in a Faster R-CNN object detection model.

    Key Features:
    - Loads pretrained Swin-T and extracts its hierarchical feature stages.
    - Selects a configurable number of intermediate feature maps to use as backbone outputs.
    - Applies 1x1 convolutions to unify feature map channel dimensions to 256 (required by RPN and ROI heads).
    - Converts Swin feature maps from (N, H, W, C) to (N, C, H, W) for compatibility with standard CNN-based detection heads.
    - Returns a dictionary of processed feature maps used as input to the RPN and detection heads.

    Args:
    - stages (int): Number of Swin-T stages to use (1 to 4).

    Output:
    - A dict with selected feature maps, e.g., {'1': tensor, '2': tensor, ...}, each shaped (N, 256, H, W).
    """
    def __init__(self, stages=1):
        super().__init__()

        assert 1 <= stages <= 4, "stages must be between 1 and 4"
        self.stages = stages

        # Load Pretrained Swin-T
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).features

        # Swin-T output channels by stage index (each stage has 2 parts)
        self.out_channels_list = [96, 96, 192, 192, 384, 384, 768, 768]

        # Use stages 1 to 4 → indices: 0–1, 2–3, 4–5, 6–7 → pick even indices
        selected_indices = [2 * i for i in range(stages)]
        self.selected_indices = [i + 1 for i in selected_indices]  # use corresponding feature output index

        # Create 1x1 conv layers to map each stage output to 256 channels
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(self.out_channels_list[idx], 256, kernel_size=1)
            for idx in selected_indices
        ])

        self.out_channels = 256  # Unified output channels

    def forward(self, x):
        """
        Forward pass through the Swin Transformer backbone, extracting features
        from specified stages and applying 1x1 convolutions to standardize output channels.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, C, H, W).

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping stage indices ('1', '2', ...) 
            to feature maps of shape (N, C, H, W), where C is standardized by the conv layers.

        Process:
            1. Iterates over Swin stages and collects features at selected stages.
            2. Converts feature maps from (N, H, W, C) to (N, C, H, W).
            3. Applies 1x1 convolutions to each selected feature map.
            4. Renames the keys to normalized string indices ('1', '2', ..., etc.).
        """
        features = {}
        for i, stage in enumerate(self.backbone):
            x = stage(x)
            if i in self.selected_indices:
                features[f'{i}'] = x

        # Apply 1x1 convs and permute feature maps to (N, C, H, W)
        for conv_idx, i in enumerate(self.selected_indices):
            features[f'{i}'] = features[f'{i}'].permute(0, 3, 1, 2).contiguous()
            features[f'{i}'] = self.conv_layers[conv_idx](features[f'{i}'])

        # Create filtered dict with keys normalized to '1', '2', ..., 'n'
        filtered_features = {
            str(j + 1): features[f'{i}']
            for j, i in enumerate(self.selected_indices)
        }

        return filtered_features

class FasterRcnn(nn.Module):
  def __init__(self,backbone,num_classes,device):
    """
        Initializes the Faster R-CNN model with a given backbone and number of classes.

        Args:
            backbone (nn.Module): Feature extractor network (e.g., a CNN or transformer).
                                 It must have an `out_channels` attribute defining the number
                                 of output feature channels.
            num_classes (int): Number of target object classes (including background if applicable).
        """
    
    super(FasterRcnn,self).__init__()

    self.num_classes = num_classes

    # Anchor configuration:
    #   - scales: how big the anchor boxes are
    #   - aspect_ratios: shape ratios (width / height)
    self.scales=[128,256,512]
    self.aspect_ratios=[0.5,1.5,2]
    self.num_anchors=len(self.scales)*len(self.aspect_ratios)

    # Normalization statistics (ImageNet-standard for natural images)
    self.image_mean = [0.485, 0.456, 0.406]
    self.image_std = [0.229, 0.224, 0.225]

    # Image resizing bounds
    self.min_size=600
    self.max_size=1000

    # Backbone network (e.g., ResNet, Swin Transformer)
    # It is expected to have an attribute: out_channels
    self.backbone = backbone

    # Region Proposal Network (RPN): proposes candidate object regions
    self.rpn = RPN(backbone.out_channels,anchors=self.num_anchors)

    # Detection head: classifies and regresses proposals
    self.head = DetectionHead(in_channels=backbone.out_channels,num_classes=num_classes,roi_output_size=7).to(torch.float64)

    # RoI Pooling module to extract fixed-size features from proposals
    # MultiScaleRoIAlign is used for feature alignment from multiscale maps

    self.roi_pooler = MultiScaleRoIAlign(
            featmap_names=['1'],  # Use specific feature map level (e.g., "P1" from FPN or "1" from the stages in a Swin-Transformer)
            output_size=7,        # Resize region to 7×7
            sampling_ratio=2      # Number of samples per bin in pooling
        ).to(torch.float64)

    self.device=device

  def normalize_resize_image_and_boxes(self, image, bboxes=None):
    """
    Normalizes and resizes an input image (and optionally its bounding boxes)
    to fit within a predefined size range while maintaining aspect ratio.

    This is commonly used as a preprocessing step before feeding data into a neural network.

    Args:
        image (Tensor): Input image of shape (C, H, W), with values in range [0, 255] or [0.0, 1.0].
        bboxes (Tensor, optional): Bounding boxes of shape (N, 1, 4) in (xmin, ymin, xmax, ymax) format.

    Returns:
        Tuple:
            image (Tensor): Normalized and resized image.
            bboxes (Tensor or None): Resized bounding boxes (if provided), same format as input.
    """

    dtype, device = image.dtype, image.device

    # Normalize image using predefined mean and std per channel 
    mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
    std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
    image = (image - mean[:, None, None]) / std[:, None, None]
    
    # Get original height and width of the image
    h, w = image.shape[-2:]
    im_shape = torch.tensor(image.shape[-2:])
    min_size = torch.min(im_shape).to(dtype=torch.float32)
    max_size = torch.max(im_shape).to(dtype=torch.float32)

    # Calculate scale factor so that:
    # - the shorter side becomes self.min_size
    # - the longer side does not exceed self.max_size
    scale = torch.min(float(self.min_size) / min_size, float(self.max_size) / max_size)
    
    # Compute new height and width using the scale
    new_h = int(h * scale)
    new_w = int(w * scale)
    new_h = max(self.min_size, min(new_h, self.max_size))
    new_w = max(self.min_size, min(new_w, self.max_size))

    # Resize the image using bilinear interpolation
    image = torch.nn.functional.interpolate(
        image,
        size=(new_h,new_w),
        mode="bilinear",
        align_corners=False,
    )

    # If bounding boxes are provided, scale them to match resized image
    if bboxes is not None:
        # Resize boxes by
        ratios = [
            torch.tensor(s, dtype=torch.float64, device=bboxes.device)
            / torch.tensor(s_orig, dtype=torch.float64, device=bboxes.device)
            for s, s_orig in zip(image.shape[-2:], (h, w))
        ]
        ratio_height, ratio_width = ratios

        # Adjust each coordinate of the bounding boxes
        xmin, ymin, xmax, ymax = bboxes.unbind(2)
        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)
        
    return image, bboxes

  def generate_anchors(self, image, feats):
    """
    Generate anchors for multi-scale feature maps extracted from the backbone (e.g., Swin Transformer).
    
    Anchors are generated at different scales and aspect ratios across all feature map levels.
    These serve as candidate regions for object detection, to be refined during training/inference.

    Args:
        image (Tensor): Input image tensor of shape (N, C, H, W), used to derive spatial dimensions.
        feats (List[Tensor]): List of feature map tensors from different pyramid levels.
                              Each tensor is of shape (N, C_l, H_l, W_l), where H_l and W_l vary by level.

    Returns:
        Tensor: Combined anchor boxes from all feature map levels, shape (total_anchors, 4),
                with coordinates in (x1, y1, x2, y2) format.
    """

    image_h, image_w = image.shape[-2:]
    all_anchors = []

    # Convert predefined anchor scales and aspect ratios to tensors
    scales = torch.as_tensor(self.scales, dtype=image.dtype, device=image.device)
    aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=image.dtype, device=image.device)

    # Compute h and w ratios for aspect ratios
    h_ratios = torch.sqrt(aspect_ratios)
    w_ratios = 1 / h_ratios

    # Compute anchor widths and heights
    ws = (w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h_ratios[:, None] * scales[None, :]).view(-1)

    # Generate base anchors (zero-centered)
    base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
    base_anchors = base_anchors.round()

    # Iterate through each feature level
    for feat in feats:
        grid_h, grid_w = feat.shape[-2:]

        # Calculate stride (i.e., spatial resolution ratio) for current feature level
        stride_h = torch.tensor(image_h // grid_h, dtype=torch.int64, device=feat.device)
        stride_w = torch.tensor(image_w // grid_w, dtype=torch.int64, device=feat.device)

        # Generate shift positions along x and y directions
        shifts_x = torch.arange(0, grid_w, dtype=torch.int32, device=feat.device) * stride_w
        shifts_y = torch.arange(0, grid_h, dtype=torch.int32, device=feat.device) * stride_h

        # Create grid of shifts using meshgrid
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)
        
        # Combine shifts into (x1, y1, x2, y2) format (broadcasted later)
        shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1)

        # Generate anchors by adding shifts to the base anchors
        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4))
        anchors = anchors.reshape(-1, 4)

        # Append to list
        all_anchors.append(anchors)

    # Concatenate anchors from all levels
    all_anchors = torch.cat(all_anchors, dim=0)

    return all_anchors

  def clamp_boxes(self,boxes,image_shape):
    """
    Clamps the bounding box coordinates to be within the image dimensions.

    This function ensures that all bounding boxes do not exceed the boundaries
    of the image. It's especially important after applying predicted box deltas,
    which may produce coordinates outside the valid range.

    Args:
        boxes (Tensor): Tensor of shape (..., 4) containing box coordinates in the format (x1, y1, x2, y2).
        image_shape (List[Tuple[int, int]]): Image shape, typically as ((height, width),).
    
    Returns:
        Tensor: Clamped boxes with the same shape as input, where all coordinates are within image bounds.
    """
    # Extract individual box coordinates
    x1=boxes[...,0]
    y1=boxes[...,1]
    x2=boxes[...,2]
    y2=boxes[...,3]

    # Extract image height and width
    max_height,max_width=image_shape[0][0],image_shape[0][1]

    # Clamp coordinates to image boundaries
    x1=x1.clamp(min=0,max=max_width)
    x2=x2.clamp(min=0,max=max_width)

    y1=y1.clamp(min=0,max=max_height)
    y2=y2.clamp(min=0,max=max_height)

    # Concatenate clamped coordinates back into box format
    boxes=torch.cat((
        x1[...,None],
        y1[...,None],
        x2[...,None],
        y2[...,None],
    ),dim=-1)
    return boxes


  def filter_proposals(self,proposals,objectness_logits,image_shape):
    """
    Filters region proposals based on objectness scores, size, and Non-Maximum Suppression (NMS).

    This function is typically used in RPNs (Region Proposal Networks) to refine raw anchor proposals:
    - Keeps top-N proposals based on foreground scores.
    - Removes boxes that are too small or outside image boundaries.
    - Applies NMS to suppress overlapping proposals.
    - Returns top-scoring proposals and their corresponding scores.

    Args:
        proposals (Tensor): Tensor of shape (num_proposals, 4) with coordinates (x1, y1, x2, y2).
        objectness_logits (Tensor): Objectness scores (before sigmoid), shape (num_proposals, 1 or num_proposals).
        image_shape (Tuple[int, int]): The (height, width) of the input image.

    Returns:
        proposals (Tensor): Filtered proposals after NMS and size constraints, shape (N, 4).
        objectness_logits (Tensor): Corresponding scores of filtered proposals, shape (N,).
    """

    # Flatten the bbox deltas to match the anchors, and get the foreground (object) value from the clasification logits
    objectness_logits=objectness_logits.reshape(-1)

    # Convert logits to probabilities using sigmoid 
    objectness_logits = torch.sigmoid(objectness_logits)

    # Keep top 5000 proposals with the highest objectness scores
    _,top_n_idx=objectness_logits.topk(5000)
    objectness_logits=objectness_logits[top_n_idx]
    proposals=proposals[top_n_idx]

    # Ensure proposals are within image boundaries
    proposals=self.clamp_boxes(proposals,image_shape)

    # Filter out proposals that are too small (less than 16 pixels in width or height)
    min_size = 16
    ws, hs = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = torch.where(keep)[0]
    proposals = proposals[keep]
    objectness_logits = objectness_logits[keep]

    # Apply Non-Maximum Suppression (NMS) to remove highly overlapping boxes
    keep_mask = torch.zeros_like(objectness_logits, dtype=torch.bool)
    keep = nms(proposals, objectness_logits, 0.7) # IoU threshold = 0.7
    keep_mask[keep]=True

    # Sort post-NMS proposals by score in descending order
    post_nms = keep[objectness_logits[keep].sort(descending=True)[1]]

    # Keep only the top 2000 proposals
    proposals = proposals[post_nms[:2000]]
    objectness_logits=objectness_logits[post_nms[:2000]]

    return proposals,objectness_logits

  def assign_targets_to_anchors(self, proposals, gt_boxes):
    """
    Assign ground truth boxes to anchor proposals using IoU-based matching.

    This method follows standard anchor-target assignment strategy:
    - Anchors with IoU < 0.3 are marked as background.
    - Anchors with 0.3 <= IoU < 0.7 are marked as ignored.
    - Anchors with IoU >= 0.7 are marked as positive.
    - Additionally,we ensure each ground truth box is assigned to at least one anchor.

    Args:
        proposals (Tensor): Anchors/proposal boxes of shape (num_proposals, 4).
        gt_boxes (Tensor): Ground truth boxes of shape (num_gt, 4).

    Returns:
        labels (Tensor): Tensor of shape (num_proposals,), containing:
                         - 1 for positive (foreground),
                         - 0 for negative (background),
                         - -1 for ignored anchors.
        matched_gt_boxes (Tensor): Tensor of shape (num_proposals, 4), containing
                                   matched GT boxes for each anchor.
    """

    # Compute IOU between each ground-truth box and proposal
    iou_matrix = box_iou(gt_boxes, proposals)  # Shape(gt_boxes, proposals)

    # Find the best ground-truth match for each proposal
    max_iou, matched_idxs = iou_matrix.max(dim=0)

    # Clone the indices for safe modifications
    copy = matched_idxs.clone()

    # Assign proposals with low IOU to background (-1)
    below_threshold = max_iou < 0.3
    matched_idxs[below_threshold] = -1

    # Assign proposals with medium IOU to ignored (-2)
    between_threshold = (max_iou >= 0.3) & (max_iou < 0.7)
    matched_idxs[between_threshold] = -2

    # Ensure each ground-truth box has at least one anchor
    best_anchor, _ = iou_matrix.max(dim=1)
    gt_pred = torch.where(iou_matrix == best_anchor[:, None])
    pred_to_update = gt_pred[1]
    matched_idxs[pred_to_update] = copy[pred_to_update]

    # Initialize matched ground-truth boxes with zeros
    matched_gt_boxes = torch.zeros_like(proposals)
    gt_boxes = gt_boxes.to(dtype=matched_gt_boxes.dtype)

    # Assign matched ground-truth boxes to valid anchors
    valid_idxs = matched_idxs >= 0
    matched_gt_boxes[valid_idxs] = gt_boxes[matched_idxs[valid_idxs]]

    # Assign labels: 1 for foreground, 0 for background, -1 for ignored
    labels = matched_idxs >= 0
    labels = labels.to(dtype=torch.float32)

    background_anchors = matched_idxs == -1
    labels[background_anchors] = 0.0

    ignored_anchors = matched_idxs == -2
    labels[ignored_anchors] = -1.0

    return labels, matched_gt_boxes


  def assign_targets_to_proposals(self,proposals,gt_boxes,gt_labels):
    """
    Assigns ground truth boxes and labels to each proposal based on IoU overlap.
    
    For each proposal, this method:
    1. Computes the IoU with all ground truth boxes.
    2. Assigns the ground truth box with the highest IoU (above a threshold) to the proposal.
    3. Labels proposals as:
       - Positive: IoU >= 0.5
       - Background: 0.0 <= IoU < 0.5
       - Ignored: IoU < 0.0 
    
    Args:
        proposals (Tensor): Tensor of shape (num_proposals, 4), proposal boxes in (x1, y1, x2, y2) format.
        gt_boxes (Tensor): Tensor of shape (num_gt, 4), ground truth boxes.
        gt_labels (Tensor): Tensor of shape (num_gt,), containing the class label for each ground truth box.

    Returns:
        labels (Tensor): Tensor of shape (num_proposals,), where:
                         - positive labels correspond to object classes,
                         - 0 indicates background,
                         - -1 indicates ignored proposals.
        matched_gt_boxes (Tensor): Tensor of shape (num_proposals, 4), matched ground truth box for each proposal.
    """
    # Compute IoU between each proposal and each ground truth box
    iou_matrix = box_iou(gt_boxes, proposals)  # Shape: (gt_boxes,proposals)
    
    # For each proposal, find the max IoU and index of the best-matching GT box
    max_iou, matched_idxs = iou_matrix.max(dim=0) # Shape : (proposals, )

    # Identify background (low IoU) and ignored proposals
    background_proposals=(max_iou<0.5) & (max_iou>=0.0)
    ignored_proposals=max_iou<0.0

    # Mark background and ignored proposals
    matched_idxs[background_proposals]=-1
    matched_idxs[ignored_proposals]=-2

    # Clamp matched indices to ensure they are valid (for indexing GT boxes)
    matched_gt_boxes=gt_boxes[matched_idxs.clamp(min=0)]

    # Assign labels based on matched GT boxes
    labels=gt_labels[matched_idxs.clamp(min=0)]
    labels=labels.to(dtype=torch.int64)
    
    # Background and ignored proposals get special labels
    labels[background_proposals]=0
    labels[ignored_proposals]=-1

    return labels,matched_gt_boxes

  def boxes_to_transformation_targets(self,ground_truth_boxes, anchors_or_proposals):
    """
    Computes the transformation (regression) targets between anchors/proposals and ground truth boxes.
    
    These targets represent how much an anchor/proposal needs to shift and scale 
    in order to match its assigned ground truth box, using the common format (tx, ty, tw, th).

    Args:
        ground_truth_boxes (Tensor): Tensor of shape (N, 4) containing ground truth boxes 
                                     in (x1, y1, x2, y2) format, matched to anchors/proposals.
        anchors_or_proposals (Tensor): Tensor of shape (N, 4) containing anchor or proposal boxes 
                                       in (x1, y1, x2, y2) format.

    Returns:
        Tensor: Regression targets of shape (N, 4), where each target is (tx, ty, tw, th):
            - tx, ty: normalized center offsets
            - tw, th: log-space width and height scaling factors
    """

    # Compute width, height, and center (x, y) for anchors/proposals
    widths = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    heights = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * widths
    center_y = anchors_or_proposals[:, 1] + 0.5 * heights

    # Compute width, height, and center (x, y) for ground truth boxes
    gt_widths = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
    gt_heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
    gt_center_x = ground_truth_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = ground_truth_boxes[:, 1] + 0.5 * gt_heights

    # Compute normalized offsets and scale changes
    targets_dx = (gt_center_x - center_x) / widths
    targets_dy = (gt_center_y - center_y) / heights
    targets_dw = torch.log(gt_widths / widths)
    targets_dh = torch.log(gt_heights / heights)

    # Stack into a single tensor: (tx, ty, tw, th)
    regression_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return regression_targets

  def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
    """
    Filters object detection predictions by:
    1. Removing low-confidence predictions
    2. Removing very small boxes
    3. Applying Non-Maximum Suppression (NMS) per class
    4. Keeping only the top-K scoring predictions

    Args:
        pred_boxes (Tensor): Tensor of shape (N, 4) containing predicted bounding boxes.
        pred_labels (Tensor): Tensor of shape (N,) containing class labels for each box.
        pred_scores (Tensor): Tensor of shape (N,) containing confidence scores for each box.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Filtered (boxes, labels, scores)
    """

    # Remove low scoring boxes
    keep = torch.where(pred_scores > 0.05)[0]
    pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

    # Remove small boxes (less than 16 pixels in width or height)
    min_size = 16
    ws, hs = pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = torch.where(keep)[0]
    pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

    # Apply class-wise Non-Maximum Supression (NMS)
    keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
    for class_id in torch.unique(pred_labels):
        # Get indices for the current class
        curr_indices = torch.where(pred_labels == class_id)[0]
        # Apply NMS on boxes of this class
        curr_keep_indices = nms(pred_boxes[curr_indices],pred_scores[curr_indices],0.3)
        # Mark these indices as kept
        keep_mask[curr_indices[curr_keep_indices]] = True

    # Keep top-scoring predictions after NMS (max 100)
    keep_indices = torch.where(keep_mask)[0]
    post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(descending=True)[1]]
    keep = post_nms_keep_indices[:100]

    # Final filtered predictions
    pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
    return pred_boxes, pred_labels, pred_scores

  def sample_positive_negative(self,labels,positive_count,total_count):

    """
    Samples a specified number of positive and negative indices based on label values.

    Args:
        labels (Tensor): A 1D tensor of class labels (0 for background, >=1 for objects).
        positive_count (int): Desired number of positive samples to draw.
        total_count (int): Total number of samples to return (positive + negative).

    Returns:
        sampled_neg_idx_mask (BoolTensor): A boolean mask indicating sampled negative indices.
        sampled_pos_idx_mask (BoolTensor): A boolean mask indicating sampled positive indices.
    """
    # Indices where label is positive (>=1)
    positive=torch.where(labels>=1)[0]

    # Indices where label is background (==0)
    negative=torch.where(labels==0)[0]

    # Determine how many positives to sample
    num_pos=positive_count
    num_pos=min(positive.numel(),num_pos)

    # Determine how many negatives to sample to reach total_count
    num_neg=total_count-num_pos
    num_neg=min(negative.numel(),num_neg)

    # Randomly permute and select indices for positives and negatives
    perm_positive_idxs=torch.randperm(positive.numel(),device=positive.device)[:num_pos]
    perm_negative_idxs=torch.randperm(negative.numel(),device=negative.device)[:num_neg]

    # Get actual sampled indices
    pos_idxs=positive[perm_positive_idxs]
    neg_idxs=negative[perm_negative_idxs]

    # Create boolean masks for sampled indices
    sampled_pos_idx_mask=torch.zeros_like(labels,dtype=torch.bool)
    sampled_neg_idx_mask=torch.zeros_like(labels,dtype=torch.bool)

    sampled_pos_idx_mask[pos_idxs]=True
    sampled_neg_idx_mask[neg_idxs]=True

    return sampled_neg_idx_mask,sampled_pos_idx_mask

  def apply_regression_pred_in_batches(self, bbox_pred, anchors_or_proposals, batch_size=1024):
    """
    Apply bounding box regression in batches to save memory.

    Args:
        bbox_pred (torch.Tensor): Bounding box predictions of shape (N, 4).
        anchors_or_proposals (torch.Tensor): Anchor boxes or proposals of shape (N, 4).
        batch_size (int): Number of items to process per batch.

    Returns:
        torch.Tensor: Transformed bounding boxes of shape (N, 4).
    """
    device = bbox_pred.device
    total_boxes = bbox_pred.size(0)
    boxes = []

    for i in range(0, total_boxes, batch_size):
        # Slice the current batch
        bbox_pred_batch = bbox_pred[i:i + batch_size]
        anchors_batch = anchors_or_proposals[i:i + batch_size]

        # Extract anchor dimensions
        w = anchors_batch[:, 2] - anchors_batch[:, 0]
        h = anchors_batch[:, 3] - anchors_batch[:, 1]

        # Compute anchor centers
        center_x = anchors_batch[:, 0] + 0.5 * w
        center_y = anchors_batch[:, 1] + 0.5 * h

        # Extract predictions
        dx = bbox_pred_batch[..., 0]
        dy = bbox_pred_batch[..., 1]
        dw = bbox_pred_batch[..., 2]
        dh = bbox_pred_batch[..., 3]

        # Clamp predictions for stability
        dw = torch.clamp(dw, max=math.log(1000.0 / 16))
        dh = torch.clamp(dh, max=math.log(1000.0 / 16))

        # Apply regression deltas
        pred_center_x = dx * w.unsqueeze(1) + center_x.unsqueeze(1)
        pred_center_y = dy * h.unsqueeze(1) + center_y.unsqueeze(1)
        pred_w = torch.exp(dw) * w.unsqueeze(1)
        pred_h = torch.exp(dh) * h.unsqueeze(1)

        # Convert back to corner coordinates
        pred_x1 = pred_center_x - 0.5 * pred_w
        pred_y1 = pred_center_y - 0.5 * pred_h
        pred_x2 = pred_center_x + 0.5 * pred_w
        pred_y2 = pred_center_y + 0.5 * pred_h

        batch_boxes = torch.stack((pred_x1, pred_y1, pred_x2, pred_y2), dim=1)
        boxes.append(batch_boxes)

    # Concatenate all batches to form the final output
    return torch.cat(boxes, dim=0)

  def forward(self,images,targets=None):
    """
    Forward pass of the Faster R-CNN model.

    This method processes input images (and optionally target annotations)
    through the backbone, region proposal network (RPN), and ROI heads to
    either compute training losses or generate final object detections.

    Args:
        images (List[Tensor]): A list of input images as tensors, typically shape (C, H, W).
        targets (List[Dict], optional): A list of dictionaries containing ground truth
            data for each image. Each dictionary should include:
                - 'bbox': List of bounding boxes in [x, y, w, h] format.
                - 'category_id': Corresponding class labels.

    Returns:
        If training:
            torch.Tensor: Scalar loss combining RPN and detection head losses.
        If evaluating:
            Dict[str, Tensor]: Dictionary containing detection results with keys:
                - 'boxes': Tensor of shape (N, 4) with predicted bounding boxes.
                - 'scores': Tensor of shape (N,) with confidence scores.
                - 'labels': Tensor of shape (N,) with predicted class labels.

    Workflow:
        1. Normalize and resize input images (and boxes if training).
        2. Extract feature maps using the backbone.
        3. Generate anchors and predict region proposals with RPN.
        4. Filter proposals and compute RPN classification + regression loss (if training).
        5. Sample proposals and assign ground truth for ROI heads (if training).
        6. Extract pooled features for each proposal using ROI pooling.
        7. Predict classification logits and refined bounding boxes.
        8. Compute ROI classification and regression loss (if training).
        9. During inference, decode predictions and apply score thresholding/NMS.

    Notes:
        - Converts feature maps and proposals to float64 before ROI pooling due to
          MultiScaleRoiAlign precision requirements.
        - Uses smooth L1 loss for localization and cross-entropy/binary cross-entropy
          for classification.
        - Assumes a single feature level is used; support for multi-level FPN/Swin-Transformers is implemented, but the changes need to be made manually (Instructions can be found in comments where the changes need to be made).
    """
    if self.training:
      
      # Extract ground truth bounding boxes and labels from the target
      gt_boxes = torch.tensor([t['bbox'] for t in targets], device=self.device).view(-1,len(targets), 4)
      gt_boxes=box_convert(gt_boxes,'xywh','xyxy')

      # Extract category IDs for ground truth labels
      gt_labels = torch.tensor([t['category_id'] for t in targets], device=self.device).view(-1,len(targets))

      # Normalize and resize images and ground truth boxes
      images,gt_boxes=self.normalize_resize_image_and_boxes(images,gt_boxes)
    else:
      # For inference (not training), only normalize and resize images (no ground truth)
      images,_=self.normalize_resize_image_and_boxes(images)

    # Get feature maps from the backbone
    featmaps = self.backbone(images)
    # This block sets up the feature maps for stage 1; for using more stages, update the range for stages 1-4, up to (1,5)
    features=([featmaps[f'{i}'] for i in range(1,2)])

    # Collect image sizes (height, width) for each image in the batch
    image_sizes = [(image.shape[1], image.shape[2]) for image in images]
    image_list = ImageList(images, image_sizes) # Create an ImageList to handle varying image sizes

    # Generate anchors (bounding box proposals) based on the image size and feature maps
    anchors=self.generate_anchors(image=images,feats=features)


    # Generate proposals using Region Proposal Network (RPN) by applying the regression predictions
    logits,bbox_deltas = self.rpn(features) # logits are class scores, bbox_deltas are bounding box offsets
    proposals = self.apply_regression_pred_in_batches(
          bbox_deltas.detach().reshape(-1, 1, 4), # Apply the bbox deltas (reshaped for batch processing)
          anchors)
    proposals = proposals.reshape(proposals.size(0), 4) # Reshape proposals to match expected format

    # Filter the proposals to remove low-quality ones based on their logits (scores)
    proposals,scores=self.filter_proposals(proposals,logits.detach(),image_list.image_sizes)
    rpn_ouput={'proposals':proposals,'scores':scores}

    # If in training mode, match the anchors to the ground truth labels and prepare for RPN loss calculation
    if self.training:
      labels_for_anchors,matched_gt_boxes=self.assign_targets_to_anchors(anchors,gt_boxes[0])

      # Prepare targets for the localization (bounding box regression) task
      regression_targets=self.boxes_to_transformation_targets(anchors,matched_gt_boxes)

      # Sample positive and negative examples for RPN loss computation
      sampled_neg_idx_mask,sampled_pos_idx_mask=self.sample_positive_negative(labels_for_anchors,positive_count=128,total_count=256)
      sampled_idxs=torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]

      # Compute RPN losses (classification and localization)
      rpn_localization_loss=(nn.functional.smooth_l1_loss(bbox_deltas[sampled_pos_idx_mask],regression_targets[sampled_pos_idx_mask],beta=1/9,reduction='sum')/sampled_idxs.numel())
      rpn_cls_loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[sampled_idxs].flatten(),labels_for_anchors[sampled_idxs].flatten())

      gt_boxes[0] = gt_boxes[0].to(proposals.dtype) # Ensure gt_boxes are in the same dtype as proposals for MultiROIScaleAlign

    # Convert proposals and feature maps to float64 to match MultiScaleRoiAlign's requirements (for precise computations)
    
    # Uncomment additional featmaps if using more than stage 1 (e.g., featmaps['2'], featmaps['3'], featmaps['4'])

    proposals=proposals.to(torch.float64)
    featmaps['1'] = featmaps['1'].to(proposals.dtype)
    #featmaps['2'] = featmaps['2'].to(proposals.dtype)
    #featmaps['3'] = featmaps['3'].to(proposals.dtype)
    #featmaps['4'] = featmaps['4'].to(proposals.dtype)

    # During training, include ground truth boxes with proposals for further processing
    if self.training and targets is not None:
      proposals=torch.cat([proposals,gt_boxes[0]],dim=0)

      # Assign targets (labels and ground truth boxes) to the proposals
      labels,matched_gt_boxes_for_proposals=self.assign_targets_to_proposals(proposals,gt_boxes[0],gt_labels[0])

      # Sample positive and negative proposals
      sampled_neg_idx_mask,sampled_pos_idx_mask=self.sample_positive_negative(labels,32,128)

      sampled_idxs=torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
      proposals=proposals[sampled_idxs]
      labels=labels[sampled_idxs]
      matched_gt_boxes_for_proposals=matched_gt_boxes_for_proposals[sampled_idxs]

      # Prepare targets for localization
      regression_targets=self.boxes_to_transformation_targets(matched_gt_boxes_for_proposals,proposals)

    # Perform ROI (Region of Interest) Pooling to extract pooled feature maps for each proposal
    pooled_features=self.roi_pooler(featmaps,[proposals],image_list.image_sizes)
    
    # Pass pooled features through the detection head to get class logits and bbox predictions
    cls_logits,bbox_pred = self.head(pooled_features)
    
    # Reshape bbox predictions and compute the output format
    num_boxes,num_classes=cls_logits.shape
    bbox_pred=bbox_pred.reshape(num_boxes,num_classes,4)
    frcnn_output={}

    # If in training mode, compute the final losses (classification and localization)
    if self.training and targets is not None:

      detection_cls_loss=torch.nn.functional.cross_entropy(cls_logits,labels)

      # Compute localization (bounding box regression) loss for positive proposals
      fg_proposal_idxs=torch.where(labels>0)[0]

      fg_class_labels=labels[fg_proposal_idxs]

      detection_localization_loss=torch.nn.functional.smooth_l1_loss(
          bbox_pred[fg_proposal_idxs,fg_class_labels],
          regression_targets[fg_proposal_idxs],
          beta=1/9,
          reduction='sum'
      )
      detection_localization_loss=detection_localization_loss/labels.numel()
      
      # Combine all losses to get the total loss
      total_loss=rpn_cls_loss+rpn_localization_loss+detection_cls_loss+detection_localization_loss

      return total_loss

    else:
      # During inference (not training), apply regression to get final predicted boxes
      pred_boxes=self.apply_regression_pred_in_batches(bbox_pred,proposals)
      pred_boxes=pred_boxes.permute(0,2,1) # Reorganize boxes to correct shape for further processing

      # Apply softmax to classification logits to get predicted probabilities
      pred_scores=torch.nn.functional.softmax(cls_logits,dim=-1)

      # Clamp predicted boxes to image size limits (avoid out-of-bounds predictions)
      pred_boxes=self.clamp_boxes(pred_boxes,image_list.image_sizes)

      # Prepare labels (predicted class ids) and scores for each proposal
      pred_labels=torch.arange(num_classes,device=self.device)
      pred_labels=pred_labels.view(1,-1).expand_as(pred_scores)

      # Remove predictions with the background label
      pred_boxes = pred_boxes[:, 1:]
      pred_scores = pred_scores[:, 1:]
      pred_labels = pred_labels[:, 1:]


      # pred_boxes -> (number_proposals, num_classes-1, 4)
      # pred_scores -> (number_proposals, num_classes-1)
      # pred_labels -> (number_proposals, num_classes-1)

      # Batch everything, by making every class prediction be a separate instance
      pred_boxes = pred_boxes.reshape(-1, 4)
      pred_scores = pred_scores.reshape(-1)
      pred_labels = pred_labels.reshape(-1)

      # Filter predictions based on confidence scores
      pred_boxes, pred_labels, pred_scores = self.filter_predictions(pred_boxes, pred_labels, pred_scores)
      
      frcnn_output['boxes'] = pred_boxes
      frcnn_output['scores'] = pred_scores
      frcnn_output['labels'] = pred_labels

      return frcnn_output