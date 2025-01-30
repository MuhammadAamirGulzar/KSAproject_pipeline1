import numpy as np
import torch
import torch.multiprocessing
from tqdm import tqdm

from ..get_encoder import get_encoder

torch.multiprocessing.set_sharing_strategy("file_system")

@torch.no_grad()
def extract_patch_features_from_dataloader(model, dataloader):
    """Uses model to extract features+labels from images iterated over the dataloader.

    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.

    Returns:
        dict: Dictionary object that contains (1) [N x D]-dim np.array of feature embeddings, and (2) [N x 1]-dim np.array of labels

    """
    all_embeddings, all_labels = [], []
    batch_size = dataloader.batch_size
    device = next(model.parameters())[0].device

    for batch_idx, (batch, target) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        remaining = batch.shape[0]
        if remaining != batch_size:
            _ = torch.zeros((batch_size - remaining,) + batch.shape[1:]).type(
                batch.type()
            )
            batch = torch.vstack([batch, _])

        batch = batch.to(device)
        with torch.inference_mode():
            embeddings = model(batch).detach().cpu()[:remaining, :].cpu()
            labels = target.numpy()[:remaining]
            assert not torch.isnan(embeddings).any()

        all_embeddings.append(embeddings)
        all_labels.append(labels)

    asset_dict = {
        "embeddings": np.vstack(all_embeddings).astype(np.float32),
        "labels": np.concatenate(all_labels),
    }

    return asset_dict


@torch.no_grad()
def extract_fivecrop_patch_features_from_dataloader(model, dataloader):
    """Uses model to extract features+labels from images iterated over the dataloader.
    In this loader the input is coming after passing through pytorch FiveCrop function. It gives output without collat_fn as  torch.Size([64, 5, 3, 224, 224])

    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.

    Returns:
        dict: Dictionary object that contains (1) [N x D]-dim np.array of feature embeddings, and (2) [N x 1]-dim np.array of labels

    """
    all_embeddings, all_labels = [], []
    batch_size = dataloader.batch_size
    device = next(model.parameters()).device

    for batch_idx, (batch, target) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        batch_size, num_crops, c, h, w = batch.size()
        
        # Flatten the crops into individual images
        batch = batch.view(-1, c, h, w)
        target = target.repeat_interleave(num_crops)
        
        batch = batch.to(device)
        with torch.inference_mode():
            embeddings = model(batch).detach().cpu()
            embeddings = embeddings.view(batch_size, num_crops, -1)
            print(f'flatten embedding dimension before aggregation{embeddings.shape}')
            embeddings = embeddings.mean(dim=1)  # Aggregate features by averaging
            print(f'flatten embedding dimension after mean avg {embeddings.shape}')
            print(f'Targets/labels before aggregation{target.shape}')
            labels = target[::num_crops].numpy()  # Take one label per original image
            print(f'Targets/labels after aggregation{labels.shape}')
            assert not torch.isnan(embeddings).any()

        all_embeddings.append(embeddings)
        all_labels.append(labels)

    asset_dict = {
        "embeddings": np.vstack(all_embeddings).astype(np.float32),
        "labels": np.concatenate(all_labels),
    }

    return asset_dict


@torch.no_grad()
def extract_fivecrop_collated_patch_features_from_dataloader(model, dataloader):
    """Uses model to extract features+labels from images iterated over the dataloader.
        In this loader the input is coming after passing through pytorch FiveCrop function. It gives output after collat_fn as  torch.Size([320, 3, 224, 224])

    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.

    Returns:
        dict: Dictionary object that contains (1) [N x D]-dim np.array of feature embeddings, and (2) [N x 1]-dim np.array of labels

    """
    all_embeddings, all_labels = [], []
    batch_size = dataloader.batch_size
    device = next(model.parameters()).device

    for batch_idx, (batch, target) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        num_crops = 5  # Since we are using FiveCrop
        actual_batch_size = batch_size // num_crops
        
        # Reshape the batch to separate crops
        batch = batch.view(actual_batch_size, num_crops, 3, 224, 224)
        target = target.view(actual_batch_size, num_crops)
        
        # Flatten the crops into individual images
        batch = batch.view(-1, 3, 224, 224)
        target = target[:, 0]  # Use the first label for each set of crops
        
        batch = batch.to(device)
        with torch.inference_mode():
            embeddings = model(batch).detach().cpu()
            embeddings = embeddings.view(actual_batch_size, num_crops, -1)
            embeddings = embeddings.mean(dim=1)  # Aggregate features by averaging
            assert not torch.isnan(embeddings).any()

        all_embeddings.append(embeddings)
        all_labels.append(target.numpy())

    asset_dict = {
        "embeddings": np.vstack(all_embeddings).astype(np.float32),
        "labels": np.concatenate(all_labels),
    }

    return asset_dict



@torch.no_grad()
def extract_wsi_features_with_chunks(model, dataloader, chunk_size=1000):
    """
    Extract features for each WSI by averaging the patch features for each slide, with dynamic chunking to avoid memory issues.
    
    Args:
    model: The feature extractor model.
    dataloader: DataLoader with batches of patches for each slide.
    chunk_size: Number of patches to process at a time to avoid OOM.
    
    Returns:
    asset_dict: Dictionary containing aggregated WSI features and labels.
    """
    all_embeddings = []  # Initialize as empty list
    all_labels = []  # Initialize as empty list
    device = next(model.parameters()).device
    print(f'The size of input dataloader is {len(dataloader)}')

    for batch_idx, (batch, target) in tqdm(enumerate(dataloader), total=len(dataloader)):
        num_patches, num_crops, c, h, w = batch.size()  # [num_patches, 5, 3, 224, 224]
        
        # Flatten the crops into individual images
        batch = batch.view(-1, c, h, w)
        target = target.repeat_interleave(num_crops)

        # Initialize empty list to store embeddings for this batch (WSI)
        wsi_embeddings = []

        # Determine number of patches in the current batch
        num_patches = batch.size(0)
        # print(f'No of patches in this batch including five crop is: {num_patches}')
        # Calculate number of chunks needed
        num_chunks = (num_patches + chunk_size - 1) // chunk_size
        
        # Process the batch in chunks
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, num_patches)
            chunk = batch[start_idx:end_idx].to(device)  # Take a chunk of patches
            # print(f'The selected chunk shape is: {chunk.shape}')
            
            with torch.inference_mode():
                chunk_embeddings = model(chunk).detach().cpu()  # Extract features for the chunk
                # print(f'The shape of chunk embeddings from model : {chunk_embeddings.shape}')
                chunk_embeddings = chunk_embeddings.view(-1, num_crops, chunk_embeddings.size(-1))  # Reshape to [chunk_size, num_crops, embedding_dim]
                # print(f'After Reshape the shape of chunk embeddings: {chunk_embeddings.shape}')
                # Mean across the 5 crops
                chunk_embeddings = chunk_embeddings.mean(dim=1)  # [chunk_size, embedding_dim]
                wsi_embeddings.append(chunk_embeddings)  # Store the chunk's embeddings
        
        # Concatenate embeddings for all chunks in this WSI
        wsi_embeddings = torch.cat(wsi_embeddings, dim=0)  # [num_patches, embedding_dim]
        # print(f'Shape of concatenated WSI embeddings: {wsi_embeddings.shape}')
        # Mean across all patches in the WSI
        slide_embedding = wsi_embeddings.mean(dim=0)  # [embedding_dim]
        # print(f'Shape of WSI after averaging {slide_embedding.shape}')
        # Take one label for the WSI
        wsi_label = target[0].item()

        all_embeddings.append(slide_embedding.numpy())
        all_labels.append(wsi_label)

    # Stack the embeddings and labels
    asset_dict = {
        "embeddings": np.stack(all_embeddings).astype(np.float32),  # [num_slides, embedding_dim]
        "labels": np.array(all_labels),
    }

    return asset_dict
