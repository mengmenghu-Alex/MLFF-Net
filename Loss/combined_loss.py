from Loss.SSIM_loss import *
from Loss.color_loss import *
from Loss.histogram_loss import *

class combinedloss(nn.Module):
    def __init__(self, config):
        super(combinedloss, self).__init__()
        self.l1loss = nn.L1Loss().to(config['device'])

    def forward(self, out, label):

        color_loss_value = color_loss(out,label)

        ssim_loss = 1-torch.mean(ssim(out, label)) ###
        
        his_loss = histogram_matching_loss(out,label)
        if not isinstance(his_loss, torch.Tensor):
            his_loss = torch.tensor(his_loss, dtype=ssim_loss.dtype, device=ssim_loss.device)

        total_loss = ssim_loss + 10*his_loss + color_loss_value        
        return total_loss, ssim_loss, 10*his_loss, color_loss_value 
       