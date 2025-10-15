
import torch
import os
import torchvision.transforms as transforms
import time 

from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import PolynomialLR
from models import FasterRcnn,SwinBackbone

def filter_dataset(dataset):
  """
    Filters out entries from the dataset that do not contain any targets (images with no annotations. COCO has more than 80k images with no annotations).

    Args:
        dataset (Dataset): A dataset (or subset) where each item is a tuple (image, target).

    Returns:
        list: A filtered list of (image, target) pairs where targets are non-empty.
    """
  
  filtered_dataset = []
  for data in dataset:
    image, target = data

    if len(target) > 0:
        filtered_dataset.append(data)
  return filtered_dataset


def train(model, train_loader,val_loader, optimizer, best_loss,scheduler,log_dir,device,num_epochs=200):
    writer = SummaryWriter(log_dir=log_dir)
    model.train()
    for epoch in range(num_epochs):
        start_time = time.time()

        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            loss = model(images, targets)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss=running_loss/len(train_loader)

        writer.add_scalar('Loss/Train',epoch_train_loss,epoch)

        model.train() # Keep in train mode to get loss during validation
        val_loss=0.0
        with torch.no_grad():
          for images,targets in val_loader:

            images = images.to(device)
            loss = model(images,targets)
            val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)

        scheduler.step()

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Time: {epoch_time:.2f}s, Best Loss: {best_loss:.4f}")

        print(f'Best loss:{best_loss}')
        if epoch_train_loss < best_loss:
          best_loss=epoch_train_loss
          torch.save(model.state_dict(),'C:/##FACULTATE/Anul 3 Semestru 1/Practica/faster_rcnn.pth')
          print('Saved')

    return best_loss

# For Tensorboard run <tensorboard --logdir=./logs --port=6006> in the terminal (without the <>) and open http://localhost:6006 in the browser

def main():
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  # Load the full dataset from the downloaded COCO dataset for training
  dataset = CocoDetection(root='C:/##FACULTATE/Anul 3 Semestru 1/Practica/content/train2017', annFile='C:/##FACULTATE/Anul 3 Semestru 1/Practica/content/annotations/instances_train2017.json', transform=transforms.ToTensor())

  small_train_ds=torch.utils.data.Subset(dataset,range(10000)) #We pick the first 10000 images for training 
  small_val_ds=torch.utils.data.Subset(dataset,range(10001,11001)) #We pick the next 1000 images for validating 

  # Note for readers : this dataset has over 300k images, and I do not have the computing power to train on all of them, so I decided to train the model on a number of images close to what ImageNET1K or VOG2017 would have.

  small_train_ds.transform=transforms.ToTensor()
  small_val_ds.transform=transforms.ToTensor()

  # Create data loaders using the filtered datasets (only images with targets)
  small_train_loader=DataLoader(filter_dataset(small_train_ds),shuffle=False)
  small_val_loader=DataLoader(filter_dataset(small_val_ds),shuffle=False)

  # Wrap Swin-T Backbone
  swin_fpn_backbone = SwinBackbone()

  # Use in Faster R-CNN
  model = FasterRcnn(swin_fpn_backbone, num_classes=91,device=device)
  model = model.to(device)

  best_loss=float('inf')
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
  scheduler = PolynomialLR(optimizer,total_iters=200,power=0.9)
  log_dir="./logs"
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  train(model,small_train_loader,small_val_loader,optimizer,best_loss,scheduler,log_dir,device=device,num_epochs=200)

if __name__ == "__main__":
   main()
  