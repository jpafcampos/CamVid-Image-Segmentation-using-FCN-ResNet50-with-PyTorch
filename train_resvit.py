import config
import argparse
import sys

from engine import Trainer
from dataset import train_dataset, train_data_loader
from dataset import valid_dataset, valid_data_loader
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import vit
import resvit

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--resume-training', dest='resume_training',
                    required=True, help='whether to resume training or not',
                    choices=['yes', 'no'])
parser.add_argument('-p', '--model-path', dest='model_path',
                    help='path to trained model for resuming training')
args = vars(parser.parse_args())

print(f"\nSAVING CHECKPOINT EVERY {config.SAVE_EVERY} EPOCHS\n")
print(f"LOGGING EVERY {config.LOG_EVERY} EPOCHS\n")

epochs = config.EPOCHS


resnet50 = models.resnet50(pretrained=args.pretrained)
resnet50_backbone = models._utils.IntermediateLayerGetter(resnet50, {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'})
model = resvit.ResViT(pretrained_net=resnet50_backbone, num_class=32, dim=768, depth=1, heads=1, mlp_dim=1024, batch_size = config.BATCH_SIZE, trans_img_size=args.size_img//8, feat = "feat2")
model.to(config.DEVICE)

# initialie `Trainer` if resuming training
if args['resume_training'] == 'yes':
    if args['model_path'] == None:
        sys.exit('\nPLEASE PROVIDE A MODEL TO RESUME TRAINING FROM!')
    trainer = Trainer( 
    model, 
    train_data_loader, 
    train_dataset,
    valid_data_loader,
    valid_dataset,
    config.CLASSES_TO_TRAIN,
    epochs,
    config.DEVICE, 
    config.LR,
    args['resume_training'],
    model_path=args['model_path']
)

# initialie `Trainer` if training from beginning
else:
    trainer = Trainer( 
        model, 
        train_data_loader, 
        train_dataset,
        valid_data_loader,
        valid_dataset,
        config.CLASSES_TO_TRAIN,
        epochs,
        config.DEVICE, 
        config.LR,
        args['resume_training']
    )

trained_epochs = trainer.get_num_epochs()
epochs_to_train = epochs - trained_epochs

train_loss , train_mIoU, train_pix_acc = [], [], []
valid_loss , valid_mIoU, valid_pix_acc = [], [], []
for epoch in range(epochs_to_train):
    epoch = epoch+1+trained_epochs
    print(f"Epoch {epoch} of {epochs}")
    train_epoch_loss, train_epoch_mIoU, train_epoch_pixacc = trainer.fit()
    valid_epoch_loss, valid_epoch_mIoU, valid_epoch_pixacc = trainer.validate(epoch)
    train_loss.append(train_epoch_loss)
    train_mIoU.append(train_epoch_mIoU)
    train_pix_acc.append(train_epoch_pixacc)
    valid_loss.append(valid_epoch_loss)
    valid_mIoU.append(valid_epoch_mIoU)
    valid_pix_acc.append(valid_epoch_pixacc)

    if epoch % config.LOG_EVERY == 0: 
        print(f"Train Epoch Loss: {train_epoch_loss:.4f}, Train Epoch mIoU: {train_epoch_mIoU:.4f}, Train Epoch PixAcc: {train_epoch_pixacc:.4f}")
        print(f"Valid Epoch Loss: {valid_epoch_loss:.4f}, Valid Epoch mIoU: {valid_epoch_mIoU:.4f}, Valid Epoch PixAcc: {valid_epoch_pixacc:.4f}")


    # save model every 5 epochs
    if epoch % config.SAVE_EVERY == 0:
        print('SAVING MODEL')
        trainer.save_model(epoch)
        print('SAVING COMPLETE')

print('TRAINING COMPLETE')