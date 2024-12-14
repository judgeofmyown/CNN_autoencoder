import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as Datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import os

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Running on the GPU')
else:
    device = torch.device('cpu')
    print('Running on the CPU')

training_set = Datasets.CIFAR10(root="./", download=True,
                                transform=transforms.ToTensor())

validation_set = Datasets.CIFAR10(root="./", download=True, train=False,
                                  transform=transforms.ToTensor())


def extract_each_class(dataset):
  images=[]
  ITERATE = True
  i=0
  j=0

  while ITERATE:
    for label in dataset.targets:
      if label == j:
        images.append(dataset.data[i])
        i+=1
        j+=1
        if j==10:
          ITERATE = False
      else:
        i+=1
  return images

training_images = [x for x in training_set.data]
validation_images = [x for x in validation_set.data]

test_images = extract_each_class(validation_set)

class CustomCIFAR(Dataset):
  def __init__(self, data, transforms=None):
    self.data=data
    self.transforms=transforms
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    image = self.data[idx]
    if self.transforms != None:
      image = self.transforms(image)
    return image

training_data = CustomCIFAR(training_images, transforms=transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
validation_data = CustomCIFAR(validation_images, transforms=transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
test_data = CustomCIFAR(test_images, transforms=transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))


class Encoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=16, latent_dim=200, act_func=nn.ReLU()):
    super().__init__()

    self.net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), # (32, 32)
        act_func,
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        act_func,
        nn.Conv2d(out_channels, 2*out_channels, 3, padding=1, stride=2), # (16, 16)
        act_func,
        nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1),
        act_func,
        nn.Conv2d(2*out_channels, 4*out_channels, 3, padding=1, stride=2), # (8, 8)
        act_func,
        nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
        act_func,
        nn.Flatten(),
        nn.Linear(4*out_channels*8*8, latent_dim),
        act_func
    )

  def forward(self, x):
    x = x.view(-1, 3, 32, 32)
    output = self.net(x)
    return output

class Decoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=16, latent_dim=200, act_func=nn.ReLU()):
    super().__init__()

    self.out_channels = out_channels

    self.linear = nn.Sequential(
        nn.Linear(latent_dim, 4*out_channels*8*8),
        act_func
    )

    self.conv = nn.Sequential(
        nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1), #(8, 8)
        act_func,
        nn.ConvTranspose2d(4*out_channels, 2*out_channels, 3, padding=1,
                           stride=2, output_padding=1), # (16, 16)
        act_func,
        nn.ConvTranspose2d(2*out_channels, 2*out_channels, 3, padding=1),
        act_func,
        nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1,
                           stride=2, output_padding=1), # (32, 32)
        act_func,
        nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1)
    )
  def forward(self, x):
    output = self.linear(x)
    output = output.view(-1, 4*self.out_channels, 8, 8)
    output = self.conv(output)
    return output
  
class Autoencoder(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder=encoder
    self.encoder.to(device)

    self.decoder = decoder
    self.decoder.to(device)
  
  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  
class ConvolutionalAutoencoder():
  def __init__(self, autoencoder):
    super().__init__()
    self.network = autoencoder
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
  
  def train(self, loss_function, epochs, batch_size,
            training_set, validation_set, test_set):
    log_dict = {
        'training_loss_perbatch': [],
        'validation_loss_perbatch': [],
        'visualizations': []
    }

    def init_weights(module):
      if isinstance(module, nn.Conv2d):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
      elif isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
    
    self.network.apply(init_weights)

    train_loader = DataLoader(training_set, batch_size)
    val_loader = DataLoader(validation_set, batch_size)
    test_loader = DataLoader(test_set, 10)

    self.network.train()
    self.network.to(device)

    for epoch in range(epochs):
      print(f'epoch {epoch + 1}/{epochs}')
      train_loss = []

      print('training ...')
      for image in train_loader:
        self.optimizer.zero_grad()
        
        image = image.to(device)
        output = self.network(image)

        loss = loss_function(output, image.view(-1, 3, 32, 32))
        loss.backward()
        self.optimizer.step()

        log_dict['training_loss_perbatch'].append(loss.item())
      

      print('validating ...')
      for val_images in val_loader:
        with torch.no_grad():
          val_images = val_images.to(device)
          output = self.network(val_images)
          val_loss = loss_function(output, val_images.view(-1, 3, 32, 32))
        
        log_dict['validation_loss_perbatch'].append(val_loss.item())
      

      print(f'training_loss: {round(loss.item(), 4)} validation_loss: {round(val_loss.item(), 4)}')


      for test_images in test_loader:
        test_images = test_images.to(device)
        with torch.no_grad():
          reconstructed_imgs = self.network(test_images)
        reconstructed_imgs = reconstructed_imgs.cpu()
        test_images = test_images.cpu()

        imgs = torch.stack([test_images.view(-1, 3, 32, 32), reconstructed_imgs], 
                              dim=1).flatten(0,1)
        grid = make_grid(imgs, nrow=10, normalize=True, padding=1)
        grid = grid.permute(1, 2, 0)
        plt.figure(dpi=170)
        plt.title('Original/Reconstructed')
        plt.imshow(grid)
        log_dict['visualizations'].append(grid)
        plt.axis('off')
        plt.show()
    return log_dict
  
  def autoencode(self, x):
    return self.network(x)
  
  def encode(self, x):
    encoder = self.network.encoder
    return encoder(x)
  
  def decode(self, x):
    decoder = self.network.decoder
    return decoder(x)

model = ConvolutionalAutoencoder(Autoencoder(Encoder(), Decoder()))
log_dict = model.train(nn.MSELoss(), epochs=10, batch_size=64, 
                           training_set=training_data, validation_set=validation_data,
                           test_set=test_data)

# os.makedirs("./model", exist_ok=True)
# torch.save(model, "./model/cnnAE.pth")