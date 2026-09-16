from pathlib import Path
from tqdm import tqdm

import typer

import torchvision.transforms as torchvision_T
from torchvision import datasets
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import efficientnet_b0

from simpsons_face_recognition.config import MODELS_DIR, EXTERNAL_DATA_DIR

IMG_SIZE=(224, 224)
VALID_SIZE=0.2 #20% of training set should go to validation set
BATCH_SIZE=128

LEARNING_RATE=0.0001
device='cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE=128

def normalize(IMG_SIZE, pretrained=True):
  if pretrained:
    return torchvision_T.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
  else:
    return torchvision_T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def train_transforms(IMG_SIZE, pretrained=True):
  transforms=torchvision_T.Compose([torchvision_T.Resize(IMG_SIZE), torchvision_T.RandomGrayscale(p=0.4), torchvision_T.RandomHorizontalFlip(p=0.5),
                                    torchvision_T.RandomAdjustSharpness(2, 0.5), torchvision_T.GaussianBlur((5, 9), (0.1, 5)), torchvision_T.ToTensor(), normalize(IMG_SIZE, pretrained)])
  return transforms

def valid_transforms(IMG_SIZE, pretrained=True):
  transforms=torchvision_T.Compose([torchvision_T.Resize(IMG_SIZE), torchvision_T.ToTensor(), normalize(IMG_SIZE, pretrained)])
  return transforms
  
def get_datasets(root_dir, pretrained=True):
  dataset=datasets.ImageFolder(root_dir, transform=train_transforms(IMG_SIZE))

  sz=len(dataset)
  valid_size=int(VALID_SIZE*sz)
  indices=torch.randperm(len(dataset)).tolist()
  dataset_train=Subset(dataset, indices[:-valid_size])
  dataset_valid=Subset(dataset, indices[-valid_size:])

  return dataset_train, dataset_valid, dataset.classes

def get_loaders(root_dir, pretrained=True):
  train, valid, classes=get_datasets(root_dir)
  print("# classes: ", len(classes))
  print(classes)
  train_loader=DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
  valid_loader=DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
  return train_loader, valid_loader
  

def build_model(num_classes=42):
  model=efficientnet_b0(weights='DEFAULT')

  #for finetuning, set requires_grad to True in order to train intermediate layers
  for params in model.parameters():
    params.requires_grad=True

  #modifying classifier layer to fit num_classes
  model.classifier[1]=torch.nn.Linear(in_features=1280, out_features=num_classes)
  return model

def training_step(model, loader, epoch, optimizer, criterion):
  model.train()
  training_loss=0.0
  training_num_correct=0
  counter=0
  for i, data in tqdm(enumerate(loader), desc=f'train :: epoch {str(epoch)}', total=len(loader)):
    counter+=1

    #passing training data through the model
    img, labels=data
    img=img.to(device)
    labels=labels.to(device)
    preds=model(img)

    #calculating loss
    curr_loss=criterion(preds, labels)
    training_loss+=curr_loss.item()

    #calculating accuracy
    _, preds=torch.max(preds.data, 1)
    training_num_correct+=(preds==labels).sum().item()

    #backpropagation
    curr_loss.backward()

    optimizer.step()

  #epoch statistics
  epoch_loss=training_loss/counter #mean loss
  epoch_acc=100.*(training_num_correct/len(loader.dataset))

  return epoch_loss, epoch_acc

def valid_step(model, loader, epoch, criterion):
  model.eval()
  valid_loss=0.0
  valid_correct=0
  counter=0
  with torch.no_grad():
    for i, data in tqdm(enumerate(loader), desc=f"valid :: epoch {str(epoch)}", total=len(loader)):
      counter+=1

      #run data through the model
      img, labels=data
      img=img.to(device)
      labels=labels.to(device)
      preds=model(img).detach()

      #calculate loss
      loss=criterion(preds, labels)
      valid_loss+=loss.item()

      #calculating accuracy
      _, preds=torch.max(preds.data, 1)
      valid_correct+=(preds==labels).sum().item()

  #epoch statistics
  epoch_loss=valid_loss/counter #mean loss
  epoch_acc=100.*(valid_correct/len(loader.dataset))

  return epoch_loss, epoch_acc


app = typer.Typer()

@app.command()
def main(
    features_path: Path = EXTERNAL_DATA_DIR / "simpsons_dataset" / "simpsons_dataset",
    model_path: Path = MODELS_DIR / "finetuned_efficientnet.pkl",
    epochs: int = 25
):
    train_loader, valid_loader=get_loaders(features_path)
    model=build_model()
    criterion=torch.nn.CrossEntropyLoss() #can assign weights to each of the classes
    optimizer=torch.optim.Adam(model.parameters(), lr=0.0001)
    model.to(device)

    train_loss_log, train_acc_log=[], []
    valid_loss_log, valid_acc_log=[], []
    best_acc=0.0
    torch.cuda.empty_cache()

    for epoch in range(1, epochs+1):
      tloss, tacc=training_step(model, train_loader, epoch, optimizer, criterion)
      vloss, vacc=valid_step(model, valid_loader, epoch, criterion)

      train_loss_log.append(tloss)
      train_acc_log.append(tacc)
      valid_loss_log.append(vloss)
      valid_acc_log.append(vacc)

      if vacc>=best_acc:
        torch.save(model.state_dict(), model_path)

      print('-'*50)
    
    print("Training complete")


if __name__ == "__main__":
    app()
