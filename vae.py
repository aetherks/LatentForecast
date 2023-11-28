import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from tqdm import tqdm
import copy
from arch import VAE_MLP  

#Data prep
data = np.load('sstna.npz')
print(data.keys())
data = data['arr_0']
data[np.isnan(data)] = 0
data[data>10000] = 0
data[data<-10000] = 0
print(data.shape)

train_data = []
test_data = []

for i in range(0, data.shape[0], 2400):
    train_data.append(data[i:i+1800])
    test_data.append(data[i+1800:i+2400])

train_data = np.vstack(train_data)
test_data = np.vstack(test_data)

# Compute the mean and std dev along the time dimension
mean_train = np.mean(train_data, axis=0)
std_train = np.std(train_data)

# Normalize the data
train_data = (train_data - mean_train) / std_train
test_data = (test_data - mean_train) / std_train

train_data[np.isnan(train_data)] = 0
train_data[train_data>10000] = 0
train_data[train_data<-10000] = 0

test_data[np.isnan(test_data)] = 0
test_data[test_data>10000] = 0
test_data[test_data<-10000] = 0


print(train_data.min(), train_data.max())
print(test_data.min(), test_data.max())

train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
print(f'train size: {train_data_tensor.shape}')
print(f'test size: {test_data_tensor.shape}')
# 2. NN Architecture
batch_size = 100
num_train_batches = len(train_data_tensor)//batch_size
num_test_batches = len(test_data_tensor)//batch_size

train_data_batches = torch.split(train_data_tensor, num_train_batches)
test_data_batches = torch.split(test_data_tensor, num_test_batches)

data_batches = torch.split(torch.tensor(data), 1000)

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + 0.1*KLD

# 3. Training Loop

model = VAE_MLP().cuda()

# Use the AdamW optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Define the Cosine Annealing scheduler with warm restarts
# T_0 is the number of epochs before the first restart, T_mult increases the period of restart after every restart.
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

num_epochs = 200
min_loss = 1000
for epoch in tqdm(range(num_epochs)):
    # Training
    model.train()
    train_loss = 0
    for batch_idx, batch_data in enumerate(train_data_batches):
        batch_data = batch_data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch_data)
        loss = vae_loss(recon_batch, batch_data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        scheduler.step(epoch + batch_idx / len(train_data_batches))
    train_loss /= len(train_data_batches)
    # Testing
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_data in test_data_batches:
            batch_data = batch_data.cuda()
            recon_batch, mu, logvar = model(batch_data)
            loss = vae_loss(recon_batch, batch_data, mu, logvar)
            test_loss += loss.item()
        test_loss /= len(test_data_batches)
        if min_loss>test_loss:
            min_loss = test_loss
            model_best = copy.deepcopy(model)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_data_tensor)}, Test Loss: {test_loss / len(test_data_tensor)}")
    

#Get latents for the whole dataset
latent_list = []
recons_list = []
input_list = []
for ind, data_batch in enumerate(data_batches):
    data_batch = data_batch.cuda()
    z = model.encoder(data_batch)
    x_recon, _, _ = model(data_batch)
    if ind==0:
        print(z.shape, x_recon.shape)
    latent_list.extend(z) 
    recons_list.extend(x_recon) 
    input_list.extend(data_batch) 
    
latents = torch.stack(latent_list).view(-1, z.shape[1]).detach().cpu().numpy()
recons = torch.stack(recons_list).view(-1, *x_recon.shape[1:]).detach().cpu().numpy()
inputs = torch.stack(input_list).view(-1, *data_batch.shape[1:]).detach().cpu().numpy()

print(latents.shape, recons.shape, inputs.shape)
np.save('latents', latents)
np.save('recons', recons)
np.save('inputs', inputs)

