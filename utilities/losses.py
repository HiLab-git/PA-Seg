import math
from numpy import imag
from numpy.core.fromnumeric import var
import torch
from torch import nn, zeros_like
from torch.functional import _return_counts
from utilities.focal import FocalLoss


class PartialLoss(nn.Module):
    def __init__(self, criterion):
        super(PartialLoss, self).__init__()

        self.criterion = criterion
        self.nb_classes = self.criterion.nb_classes

    def forward(self, outputs, partial_target, phase='training'):
        nb_target = outputs.shape[0]
        loss_target = 0.0
        total = 0

        for i in range(nb_target):
            partial_i = partial_target[i,...].reshape(-1)
            outputs_i = outputs[i,...].reshape(self.nb_classes, -1).unsqueeze(0)
            outputs_i = outputs_i[:,:,partial_i<self.nb_classes]

            nb_annotated = outputs_i.shape[-1]
            if nb_annotated>0:
                outputs_i= outputs_i.reshape(1,self.nb_classes,1,1,nb_annotated) # Reshape to a 5D tensor
                partial_i = partial_i[partial_i<self.nb_classes].reshape(1,1,1,1,nb_annotated) # Reshape to a 5D tensor
                loss_target += self.criterion(outputs_i, partial_i.type(torch.cuda.IntTensor), phase)
                total+=1

        if total>0:
            return loss_target/total
        else:
            return 0.0      
          
            
class DC(nn.Module):
    def __init__(self,nb_classes):
        super(DC, self).__init__()
        
        self.softmax = nn.Softmax(1)
        self.nb_classes = nb_classes

    @staticmethod 
    def onehot(gt,shape):
        shp_y = gt.shape
        gt = gt.long()
        y_onehot = torch.zeros(shape)
        y_onehot = y_onehot.to(gt.device)  ##################################
        y_onehot.scatter_(1, gt, 1)
        return y_onehot

    def reshape(self,output, target):
        batch_size = output.shape[0]

        if not all([i == j for i, j in zip(output.shape, target.shape)]):
            target = self.onehot(target, output.shape)

        target = target.permute(0,2,3,4,1)
        output = output.permute(0,2,3,4,1)
        print(target.shape,output.shape)
        return output, target


    def dice(self, output, target):
        output = self.softmax(output)
        if not all([i == j for i, j in zip(output.shape, target.shape)]):
            target = self.onehot(target, output.shape)

        sum_axis = list(range(2,len(target.shape)))

        s = (10e-20)
        intersect = torch.sum(output * target,sum_axis)
        dice = (2 * intersect) / (torch.sum(output,sum_axis) + torch.sum(target,sum_axis) + s)
        #dice shape is (batch_size, nb_classes)
        return 1.0 - dice.mean()  


    def forward(self, output, target, phase="training"):
        result = self.dice(output, target)
        return result

class CE(nn.Module):
    def __init__(self,nb_classes):
        super(CE, self).__init__()
        self.nb_classes = nb_classes
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, output, target, phase="training"):

        # Cross entropy
        output = output.permute(0,2,3,4,1).contiguous().view(-1,self.nb_classes)
        target = target.view(-1,).long()
        ce_loss = self.ce(output, target)

        result = ce_loss
        return result

class DC_CE(DC):
    def __init__(self,nb_classes):
        super(DC_CE, self).__init__(nb_classes)

        self.ce = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, output, target, phase="training"):
        # Dice term
        dc_loss = self.dice(output, target)

        # Cross entropy
        output = output.permute(0,2,3,4,1).contiguous().view(-1,self.nb_classes)
        target = target.view(-1,).long()
        ce_loss = self.ce(output, target)

        result = ce_loss + dc_loss
        return result

class DC_CE_Focal(DC):
    def __init__(self,nb_classes):
        super(DC_CE_Focal, self).__init__(nb_classes)

        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.fl = FocalLoss(reduction="none")

    def focal(self, pred, grnd, phase="training"):
        score = self.fl(pred, grnd).reshape(-1)

        if phase=="training": # class-balanced focal loss
            output = 0.0
            nb_classes = 0
            for cl in range(self.nb_classes):
                if (grnd==cl).sum().item()>0:
                    output+=score[grnd.reshape(-1)==cl].mean()
                    nb_classes+=1

            if nb_classes>0:
                return output/nb_classes
            else:
                return 0.0

        else:  # class-balanced focal loss
            return score.mean()

    def forward(self, output, target, phase="training"):
        # Dice term
        dc_loss = self.dice(output, target)

        # Focal term
        focal_loss = self.focal(output, target, phase)

        # Cross entropy
        output = output.permute(0,2,3,4,1).contiguous().view(-1,self.nb_classes)
        target = target.view(-1,).long()
        ce_loss = self.ce(output, target)

        result = ce_loss + dc_loss + focal_loss
        return result

def entropy_loss(output_soft, C=2):
    '''
    description: 
    param {output_soft: [N, C, H, W, D] the probability map after softmax layer; C: channel number.}
    return {ent: entropy loss.}
    '''    
    y1 = -1*torch.sum(output_soft*torch.log(output_soft+1e-6), dim=1) / math.log(C)
    ent = torch.mean(y1)

    return ent

class SoftCrossEntropyLoss():
   def __init__(self, weights = None):
      super().__init__()
      self.weights = weights

   def forward(self, y_hat, y):
      p = nn.functional.log_softmax(y_hat, 1)
      w_labels = self.weights*y
      loss = -(w_labels*p).sum() / (w_labels).sum()
      return loss

def entropy_loss_map(output_soft, C=2):
    ent = -1*torch.sum(output_soft * torch.log(output_soft + 1e-6), dim=1, keepdim=True) / math.log(C)
    return ent

def tv_loss(prediction):
    '''
    description: 
    param {prediction: [N, 1, H, W, D] single channel of the probability map after softmax layer or the 
    probability map after sigmoid layer.}
    return {lenght: TV loss}
    '''
    min_pool_x = nn.functional.max_pool3d(
        prediction * -1, 3, 1, 1) * -1
    contour = torch.relu(nn.functional.max_pool3d(
        min_pool_x, 3, 1, 1) - min_pool_x)
    # length
    length = torch.mean(torch.abs(contour))
    return length

class KDLoss(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(KDLoss, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = nn.functional.kl_div(nn.functional.log_softmax(out_s/self.T, dim=1),
						nn.functional.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T
		return loss

class SizeLoss(nn.Module):
    def __init__(self, margin = 0.1):
        super(SizeLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        output_counts = torch.sum(torch.softmax(output, dim=1), dim=(2, 3, 4))
        target_counts = torch.zeros_like(output_counts)
        for b in range(0, target.shape[0]):
            elements, counts = torch.unique(target[b, :, :, :, :], sorted=True, return_counts=True)
            assert torch.numel(target[b, :, :, :, :]) == torch.sum(counts)
            target_counts[b, :] = counts

        lower_bound = target_counts * (1 - self.margin)
        upper_bound = target_counts * (1 + self.margin)
        too_small = output_counts < lower_bound
        too_big = output_counts > upper_bound
        penalty_small = (output_counts - lower_bound) ** 2
        penalty_big = (output_counts - upper_bound) ** 2
        # do not consider background(i.e. channel 0)
        res = too_small.float()[:, 1:] * penalty_small[:, 1:] + too_big.float()[:, 1:] * penalty_big[:, 1:]
        loss = res / (output.shape[2] * output.shape[3] * output.shape[4])
        return loss.mean()

class TightnessPrior(nn.Module):
    def __init__(self, width = 10):
        super(TightnessPrior, self).__init__()
        self.width = int(width)

    def forward(self, output, target):
        # output.shape = [B, 2, H, W, D], target.shape = [B, 1, H, W, D]
        soft_output = torch.softmax(output, dim=1)
        # print(output.shape, target.shape)
        loss = torch.Tensor([0]).to(output.device)
        for B in range(0, output.shape[0]):
            # The projection operation can be implemented by a max operation along with each axis
            for dim in range(3):
                soft_output_2D, _ = torch.max(soft_output[B, 1, :, :, :], dim=dim)  # channel 1 for foreground
                target_2D, _ = torch.max(target[B, 0, :, :, :], dim=dim)
                # print(soft_output_2D.shape, target_2D.shape)
                x, y = torch.nonzero(target_2D, as_tuple=True)
                x_min, x_max = x.min().item(), x.max().item()
                y_min, y_max = y.min().item(), y.max().item()
                x_margin, y_margin = x_max - x_min, y_max - y_min
                # print(x_min, x_max, y_min, y_max)
                # print(x_margin, y_margin)
                # draw lines along with x axis
                for i in range(x_margin // self.width):
                    mask = torch.zeros_like(soft_output_2D)
                    mask[x_min + i * self.width : x_min + (i + 1) * self.width, y_min : y_max + 1] = 1.0
                    loss += max(0, self.width - torch.sum(soft_output_2D * mask))
                if x_margin % self.width:
                    mask = torch.zeros_like(soft_output_2D)
                    mask[x_max - x_margin % self.width : x_max + 1, y_min : y_max + 1] = 1.0
                    loss += max(0, x_margin % self.width - torch.sum(soft_output_2D * mask))
                # draw lines along with y axis
                for i in range(y_margin // self.width):
                    mask = torch.zeros_like(soft_output_2D)
                    mask[x_min : x_max + 1, y_min + i * self.width : y_min + (i + 1) * self.width] = 1.0
                    loss += max(0, self.width - torch.sum(soft_output_2D * mask))
                if y_margin % self.width:
                    mask = torch.zeros_like(soft_output_2D)
                    mask[x_min : x_max + 1, y_max - y_margin % self.width : y_max + 1] = 1.0
                    loss += max(0, y_margin % self.width - torch.sum(soft_output_2D * mask))
            return loss
                
class VarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output, input):
        # output [N, C, H, W, D], input [N, 1, H, W, D]
        soft_output = torch.softmax(output, dim=1)
        loss = 0.0
        for b in range(0, output.shape[0]):
            sub_input = input[b, 0, :, :, :]
            # mask out minimum value voxels
            mask = ((sub_input - torch.min(sub_input)) > 1e-5).float()
            # mask = 1.0
            for c in range(0, output.shape[1]):
                sub_soft_output = soft_output[b, c, :, :, :]
                if c == 0:
                    nums = torch.sum(sub_soft_output * mask)
                    mean = torch.sum(sub_input * sub_soft_output * mask) / nums
                    variance = torch.sum((sub_input - mean) ** 2 *sub_soft_output * mask) / nums
                else:
                    nums = torch.sum(sub_soft_output)
                    mean = torch.sum(sub_input * sub_soft_output) / nums
                    variance = torch.sum((sub_input - mean) ** 2 *sub_soft_output) / nums
                loss += variance
        loss = loss / output.shape[0]
        return loss

class MumfordShah_Loss(nn.Module):
    def levelsetLoss(self, output, target, penalty='l1'):
        # input size = batch x 1 (channel) x height x width
        outshape = output.shape
        tarshape = target.shape
        self.penalty = penalty
        loss = 0.0
        for ich in range(tarshape[1]):
            target_ = torch.unsqueeze(target[:, ich], 1)
            target_ = target_.expand(
                tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pcentroid = torch.sum(target_ * output, (2, 3)
                                  ) / torch.sum(output, (2, 3))
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - \
                pcentroid.expand(
                    tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss)
        return loss

    def gradientLoss2d(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if self.penalty == "l2":
            dH = dH * dH
            dW = dW * dW

        loss = torch.sum(dH) + torch.sum(dW)
        return loss

    def forward(self, image, prediction):
        loss_level = self.levelsetLoss(image, prediction)
        loss_tv = self.gradientLoss2d(image)
        return loss_level + loss_tv

class MumfordShah_Loss3D22D(MumfordShah_Loss):
    def forward(self, image, prediction):
        N, C, H, W, D = prediction.shape
        #reshape [N, C, H, W, D] tensor to [N*D, C, H, W]
        prediction = prediction.permute(0, 4, 1, 2, 3).reshape(N * D, C, H, W)
        image = image.permute(0, 4, 1, 2, 3).reshape(N * D, 1, H, W)
        loss_level = self.levelsetLoss(image, prediction)
        loss_tv = self.gradientLoss2d(image)
        return loss_level + loss_tv