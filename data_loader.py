import torch
import torch.utils.data
import torchvision.transforms as T
import data_split

def LoadData(load_dir, batchsize_train, batchsize_val, split_id, method):
    
    if method == 'guo':
        train_dataset = data_split.ImageDataset(
            data_dir = load_dir,
            split=split_id,
            mode = 'train',
            transform = T.Compose([T.ToTensor(),
                                T.Resize((256, 256)),
                                T.RandomHorizontalFlip(0.5),
                                T.RandomAffine(degrees=(-5, 5), 
                                                translate=(0.05, 0.05),
                                                scale=(0.9, 1.1)),
                                #gauss_noise_tensor
                                ]),
            temporal = False
            )
    elif method == 'sajid':
        train_dataset = data_split.ImageDataset(
            data_dir = load_dir,
            split=split_id,
            mode = 'train',
            transform = T.Compose([T.ToTensor(),
                                T.Resize((256, 256))
                                ]),
            temporal=False
        )    
    else:
        train_dataset = data_split.ImageDataset(
            data_dir = load_dir,
            split=split_id,
            mode = 'train',
            transform = T.Compose([T.ToTensor(),
                                T.Resize((256, 256))
                                ]),
            temporal=True
        )

    if method == 'yu':
        val_dataset = data_split.ImageDataset(
            data_dir = load_dir,
            split=split_id,
            mode = 'val',
            transform = T.Compose([T.ToTensor(),
                                T.Resize((256, 256))]),
            temporal=True
            )
        test_dataset = data_split.ImageDataset(
            data_dir = load_dir,
            split=split_id,
            mode = 'test',
            transform = T.Compose([T.ToTensor(),
                                T.Resize((256, 256))]),
            temporal=True
            )
    else:
        val_dataset = data_split.ImageDataset(
            data_dir = load_dir,
            split=split_id,
            mode = 'val',
            transform = T.Compose([T.ToTensor(),
                                T.Resize((256, 256))]),
            temporal=False
            )
        test_dataset = data_split.ImageDataset(
            data_dir = load_dir,
            split=split_id,
            mode = 'test',
            transform = T.Compose([T.ToTensor(),
                                T.Resize((256, 256))]),
            temporal=False
            )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train,
        shuffle=True,
        num_workers=16,
        pin_memory=True)
        
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_val,
        shuffle=True,
        num_workers=16,
        pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batchsize_val,
        shuffle=True,
        num_workers=16,
        pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def TestData(load_dir, batch_size, method):
    
    if method == 'yu':
        test_dataset = data_split.CrossDataset(
            data_dir = load_dir,
            transform = T.Compose([T.ToTensor(),
                                T.Resize((256, 256))]),
            temporal=True
            )
    
    else:
        test_dataset = data_split.CrossDataset(
            data_dir = load_dir,
            transform = T.Compose([T.ToTensor(),
                                T.Resize((256, 256))]),
            temporal=False
            )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True)
    
    return test_loader, test_dataset
