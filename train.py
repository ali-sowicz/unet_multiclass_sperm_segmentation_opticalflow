import sys
import os
from tqdm import tqdm
import datetime
import json
import torch
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter
from model import UNet, make_dataloaders, eval_net_loader, make_checkpoint_dir
from lib import plot_net_predictions
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PATH_PARAMETERS = './config.json'

def train_epoch(epoch,train_loader,criterion,optimizer,batch_size,scheduler,scaler):
    
    net.train()
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
    for i, sample_batch in enumerate(pbar):

        imgs = sample_batch['image']
        true_masks = sample_batch['mask']

        imgs = imgs.to(device)
        true_masks = true_masks.to(device)

        outputs = net(imgs)

        probs = torch.softmax(outputs, dim=1)
        masks_pred = torch.argmax(probs, dim=1)

        loss = criterion(outputs, true_masks)
        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        # save to summary
        if i%100==0:
            writer.add_scalar('train_loss_iter', 
                                  loss.item(), 
                                  i + len(train_loader) * epoch)
            writer.add_figure('predictions vs. actuals',   
                                  plot_net_predictions(imgs, true_masks, masks_pred, batch_size),    
                                  global_step = i + len(train_loader) * epoch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        optimizer.zero_grad()
        
    print(f'Epoch finished ! Loss: {epoch_loss/i:.2f}, lr:{scheduler.get_last_lr()}')



def validate_epoch(epoch,train_loader,val_loader,device):
    
    class_iou, mean_iou = eval_net_loader(net, val_loader, 3, device)
    print('Class IoU:', ' '.join(f'{x:.3f}' for x in class_iou), f'  |  Mean IoU: {mean_iou:.3f}') 
    # save to summary
    writer.add_scalar('mean_iou', mean_iou, len(train_loader) * (epoch+1))
    writer.add_scalar('Class 0', class_iou[0], len(train_loader) * (epoch+1))
    writer.add_scalar('Class 1', class_iou[1], len(train_loader) * (epoch+1))
    writer.add_scalar('Class 2', class_iou[2], len(train_loader) * (epoch+1))
    
    return mean_iou
 

def train_net(train_loader, val_loader, net, device, epochs=5, batch_size=1, lr=0.1, save_cp=True):

    print(f'''
    Starting training:
        Epochs: {epochs}
        Batch size: {batch_size}
        Learning rate: {lr}
        Training size: {len(train_loader.dataset)}
        Validation size: {len(val_loader.dataset)}
        Checkpoints: {str(save_cp)}
        Device: {str(device)}
    ''')
          
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net. parameters(),lr=lr, momentum=0.9, weight_decay=0.0005)

    # multiply learning rate by 0.1 after 30% of epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(0.3*epochs), gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()
    
    best_precision = 0
    for epoch in range(epochs):
          
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        train_epoch(epoch,train_loader,criterion,optimizer,batch_size,scheduler,scaler)

        precision = validate_epoch(epoch,train_loader,val_loader,device)
        scheduler.step()

        if save_cp and (precision>best_precision):
            state_dict = net.state_dict()
            torch.save(state_dict, dir_checkpoint+f'CP{epoch + 1}.pth')
            print('Checkpoint {} saved !'.format(epoch + 1))
            best_precision = precision
        if epoch == epochs:
            state_dict = net.state_dict()
            torch.save(state_dict, dir_checkpoint+f'CP{epoch + 1}.pth')
            
    writer.close()


def load_parameters(path):
    with open(path) as f:
        params = json.load(f)
    return params

if __name__ == '__main__':
    
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    config_data = load_parameters(PATH_PARAMETERS)
    train_param = config_data["training_parameters"]
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I+%M%p")
    dir_data = f'./data/{train_param["data_folder"]}'
    dir_checkpoint = f'./checkpoints/{train_param["data_folder"]}_b{train_param["batch_size"]}_{date_str}/'
    dir_summary = f'./runs/{train_param["data_folder"]}_b{train_param["batch_size"]}_{date_str}'
    params = {'batch_size': train_param["batch_size"], 'shuffle': True, 'num_workers': train_param["num_workers"]}

    make_checkpoint_dir(dir_checkpoint)
    writer = SummaryWriter(dir_summary)
    shutil.copy(PATH_PARAMETERS, f'{dir_summary}/config.json')
    
    val_ratio=0.1
    train_loader, val_loader =  make_dataloaders(dir_data, val_ratio, config_data, params)
    
    n_channels= config_data['models_settings']['n_channels']
    n_classes= config_data['models_settings']['n_classes']
    net = UNet(n_channels=n_channels, n_classes=n_classes)
    net.to(device)

    if train_param['load_pretrained']:
        net.load_state_dict(torch.load(train_param['load_pretrained']))
        print('Model loaded from {}'.format(train_param['load_pretrained']))
    
    # train model in parallel on multiple-GPUs
    if torch.cuda.device_count() > 1:
        print("Model training on", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net) 
    try:
        train_net(train_loader, val_loader, net, device, epochs=train_param['epochs'], batch_size=train_param["batch_size"], lr=train_param['lr'])

        
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
