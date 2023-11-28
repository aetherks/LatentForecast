import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from tqdm import tqdm
import copy
from arch import VAE_MLP2  
def reintegrate_land(recon_ocean_data, ocean_mask, original_shape):
    recon_data = np.zeros(original_shape)
    recon_data[ocean_mask] = recon_ocean_data
    return recon_data

# Assuming 'recon_ocean_data' is the output from VAE
#recon_data = reintegrate_land(recon_ocean_data, ocean_mask, flat_data.shape)



data = np.load('sstna.npz')['arr_0']
data[np.isnan(data)] = 0
data[data > 10000] = 0
data[data < -10000] = 0

# Split data into training and testing sets
train_data = []
test_data = []
for i in range(0, data.shape[0], 2400):
    train_data.append(data[i:i + 1800])
    test_data.append(data[i + 1800:i + 2400])

train_data = np.vstack(train_data)
test_data = np.vstack(test_data)

# Flatten the data and identify land points
#train_shape, test_shape. data_shape = train_data.shape, test_data.shape. data.shape
train_flat = train_data.reshape(train_shape[0], -1)
test_flat = test_data.reshape(test_shape[0], -1)
data_flat = data.reshape(data_shape[0], -1)
#train_flat_shape, test_flat_shape, data_flat_shape = train_flat.shape, test_flat.shape. data_flat.shape
print('Flattened data:')
print(train_flat_shape, test_flat_shape, data_flat_shape)
land_mask_train = (train_flat == 0)
land_mask_test = (test_flat == 0)
land_mask = (data_flat == 0)
ocean_mask = (data_flat != 0)  # True for ocean, False for land

# Filter out land points
ocean_data_train = train_flat[~land_mask_train]
ocean_data_test = test_flat[~land_mask_test]
ocean_data = data_flat[~land_mask]
# Normalize ocean data
mean_ocean_train = ocean_data_train.mean()
std_ocean_train = ocean_data_train.std()
ocean_data_train_normalized = (ocean_data_train - mean_ocean_train) / std_ocean_train
ocean_data_test_normalized = (ocean_data_test - mean_ocean_train) / std_ocean_train
ocean_data_normalized = (ocean_data - mean_ocean_train) / std_ocean_train

# Convert to PyTorch tensors
ocean_data_train_tensor = torch.tensor(ocean_data_train_normalized, dtype=torch.float32)
ocean_data_test_tensor = torch.tensor(ocean_data_test_normalized, dtype=torch.float32)
ocean_data_tensor = torch.tensor(ocean_data_normalized, dtype=torch.float32)


rint(f'train size: {ocean_data_train_tensor.shape}')
print(f'test size: {ocean_data_test_tensor.shape}')
batch_size = 100
num_train_batches = len(ocean_data_train_tensor)//batch_size
num_test_batches = len(ocean_data_test_tensor)//500

train_data_batches = torch.split(ocean_data_train_tensor, num_train_batches)
test_data_batches = torch.split(ocean_data_test_tensor, num_test_batches)
data_batches = torch.split(ocean_data_tensor, 1000)

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + 0.5*KLD


num_ocean_points = ocean_data_train_tensor.shape[1]

vae_model = VAE_MLP2(num_ocean_points).cuda()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
# T_0 is the number of epochs before the first restart, T_mult increases the period of restart after every restart.
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

# Training Loop
num_epochs = 20
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

