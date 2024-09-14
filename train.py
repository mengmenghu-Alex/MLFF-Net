import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from torch.nn import Module
from torchvision import transforms
import torch.optim as optim

import wandb
import argparse
from dataclasses import dataclass
from tqdm.autonotebook import tqdm, trange
from Utils.dataloader import myDataSet
from Utils.util import *
from Net.model import *
from Loss.combined_loss import *

os.environ["WANDB_API_KEY"]="798f6fdaf91cfb98bf18b2fae8e54212c7c303a7"
os.environ["WANDB_MODE"] = "offline"

__all__ = [
    "Trainer",
    "setup",
    "training",
]

## TODO: Update config parameter names
## TODO: remove wandb steps
## TODO: Add comments to functions

@dataclass
class Trainer:
    model: Module
    opt: torch.optim.Optimizer
    loss: Module

    @torch.enable_grad()
    def train(self, train_dataloader, model, config, val_dataloader = None):
        device = config['device']
        color_loss_lst = []
        ssim_loss_lst = []
        his_loss_lst = []
        total_loss_lst = []
        val_loss_lst = []

        for epoch in trange(0,config.num_epochs,desc = f"[Full loop]", leave = False):
            
            total_loss_tmp = 0
            color_loss_tmp = 0
            ssim_loss_tmp = 0
            his_loss_tmp = 0
            val_loss_tmp = 0
            nbs = 64
            lr_limit_max    = 1e-3 if config.optimizer_type == 'adam' else 5e-2
            lr_limit_min    = 3e-4 if config.optimizer_type == 'adam' else 5e-4
            Init_lr_fit     = min(max(config.train_batch_size / nbs * config.Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit      = min(max(config.train_batch_size / nbs * config.Init_lr * 0.01, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler(config.lr_decay_type, Init_lr_fit, Min_lr_fit, config.num_epochs)
            set_optimizer_lr(self.opt.param_groups, lr_scheduler_func, epoch)
            
            for input, label, _ in tqdm(train_dataloader, desc = f"[Train]", leave = False):
                input = input.to(device)
                label = label.to(device)

                self.model.train()

                self.opt.zero_grad()
                out = self.model(input)
                loss, ssim_loss, his_loss, color_loss = self.loss(out, label)
              

                loss.backward()
                self.opt.step()
                color_loss_tmp += color_loss.item()
                ssim_loss_tmp += ssim_loss.item()
                his_loss_tmp += his_loss.item()
                total_loss_tmp += loss.item()

            total_loss_lst.append(total_loss_tmp/len(train_dataloader))
            color_loss_lst.append(color_loss_tmp/len(train_dataloader))
            ssim_loss_lst.append(ssim_loss_tmp/len(train_dataloader))
            his_loss_lst.append(his_loss_tmp/len(train_dataloader))

            wandb.log({f"[Train] Total Loss" : total_loss_lst[epoch],
                       "[Train] SSIM Loss" : ssim_loss_lst[epoch],
                       "[Train] HIS Loss" : his_loss_lst[epoch],
                       "[Train] Color Loss" : color_loss_lst[epoch],
                    },
                      commit = True
                      )
            
            if (config.val == True) & (epoch % config.eval_steps == 0):

                val_loss = self.eval(config, val_dataloader, self.model)
                val_loss_tmp += val_loss.item()

            val_loss_lst.append(val_loss_tmp/len(val_dataloader))
            print('epoch:[{}]/[{}], val loss:{}'.format(epoch,config.num_epochs,str(val_loss_lst[epoch])))

            if epoch % config.print_freq == 0:
                print('epoch:[{}]/[{}], image loss:{}, SSIM loss:{}, HIS loss:{}, Color loss:{}'.format(epoch,config.num_epochs,str(total_loss_lst[epoch]),str(ssim_loss_lst[epoch]),str(his_loss_lst[epoch]),str(color_loss_lst[epoch])))
            if not os.path.exists(config.weights_folder):
                os.mkdir(config.weights_folder)

            if epoch % config.save_weight_freq == 0:
                torch.save(self.model, os.path.join(config.weights_folder, "ep%03d-train-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss_lst[epoch], val_loss_lst[epoch])))

            if len(val_loss_lst) <= 1 or (val_loss_lst[epoch]) <= min(val_loss_lst):
                print('Save best model to best_epoch_weights.pth')
                torch.save(self.model, os.path.join(config.weights_folder, "best_epoch_weights.pth"))         

    @torch.no_grad()
    def eval(self, config, val_dataloader, model):
        model.eval()
        for input, label, _ in tqdm(val_dataloader, desc = f"[Val]", leave = False):
            with torch.no_grad():
                input = input.to(config.device)
                label = label.to(config.device)
                out = model(input) 
                val_loss,_,_,_ = self.loss(out, label)
                
        return val_loss
    
def setup(config):
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"
    
    
    model = LEDNet().to(config["device"])

    transform = transforms.Compose([transforms.Resize((config.resize,config.resize)),transforms.ToTensor()])
    train_dataset = myDataSet(config.input_images_path,config.label_images_path,transform, True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = config.train_batch_size,shuffle = False)
    print("Train Dataset Reading Completed.")
    print(model)

    loss = combinedloss(config)
    opt = torch.optim.Adam(model.parameters(), lr=config.Init_lr*0.1)

    trainer = Trainer(model, opt, loss)

    if config.val:
        transform_val = transforms.Compose([transforms.Resize((config.resize,config.resize)),transforms.ToTensor()])
        val_dataset = myDataSet(config.val_images_path, config.val_label_images_path, transform_val, True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False)
        print("val Dataset Reading Completed.")
        return train_dataloader, val_dataloader, model, trainer
    return train_dataloader, None, model, trainer

def training(config):
    # Logging using wandb
    wandb.init(project = "LEDNet")
    wandb.config.update(config, allow_val_change = True)
    config = wandb.config
    ds_train, ds_test, model, trainer = setup(config)
    trainer.train(ds_train, model, config, ds_test)
    print("==================")
    print("Training complete!")
    print("==================")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_images_path', type=str, default="./Data/train/input/", help='path of input images(underwater images) default:./data/input/')
    parser.add_argument('--label_images_path', type=str, default="./Data/train/label/", help='path of label images(clear images) default:./data/label/')
    parser.add_argument('--val_images_path', type=str, default="./Data/val/input/", help='path of input images(underwater images) for testing default:./data/input/')
    parser.add_argument('--val_label_images_path', type = str, default="./Data/val/label/") 
    parser.add_argument('--val', default=True)
    parser.add_argument('--Init_lr', type=float, default=1e-2)
    parser.add_argument('--step_size',type=int,default=40,help="Period of learning rate decay") #
    parser.add_argument('--num_epochs', type=int, default=200)  ##
    parser.add_argument('--train_batch_size', type=int, default=64,help="default : 1")
    parser.add_argument('--val_batch_size', type=int, default=32,help="default : 1")
    parser.add_argument('--resize', type=int, default=64,help="resize images, default:resize images to 256*256") ## 对图像做了归一化
    parser.add_argument('--cuda_id', type=int, default=0,help="id of cuda device,default:0")
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--lr_decay_type', type=str, default='step')
    parser.add_argument('--print_freq', type=int, default=1)    
    parser.add_argument('--save_weight_freq', type=int, default=1)
    parser.add_argument('--weights_folder', type=str, default="./Weights/")
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--pretrain_model_path',type=str, default="")

    config = parser.parse_args()
    if not os.path.exists(config.weights_folder):
        os.mkdir(config.weights_folder)
    training(config)