import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from basic_code import load, util, networks
import sys
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description='PyTorch Frame Attention Network Training')
    parser.add_argument('--at_type', '--attention', default=1, type=int, metavar='N',
                        help= '0 is self-attention; 1 is self + relation-attention')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-f', '--fold', default=10, type=int, help='which fold used for ck+ test')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-e', '--evaluate', default=False, dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--load_model', default="/content/drive/MyDrive/MeSelves/FER_complete/emotion-FAN/model/self_relation-attention_2_100.0", type=str, help='pretrained model path')

    args = parser.parse_args()
    best_acc = 0
    at_type = ['self-attention', 'self_relation-attention'][args.at_type]
    logger.print('The attention method is {:}'.format(at_type))

    ''' Load data '''
    video_root = '/content/drive/MyDrive/MeSelves/FER_complete/evp_face'
    video_list = '/content/drive/MyDrive/MeSelves/FER_complete/EVP_txt.txt'
    batchsize_eval= 64
    val_loader = load.evp_faces_fan(video_root, video_list, 1, batchsize_eval)

    ''' Load model '''
    _structure = networks.resnet18_at(at_type=at_type)
    _parameterDir = '/content/drive/MyDrive/MeSelves/FER_complete/emotion-FAN/Resnet18_FER+_pytorch.pth.tar'
    model = load.model_parameters(_structure, _parameterDir)

    # last_model_path = "/content/drive/MyDrive/FER/emotion-FAN/model/self_relation-attention_2_96.9697" # 32.512    
    # last_model_path = "/content/drive/MyDrive/FER/emotion-FAN/model/self_relation-attention_2_100.0" # 31.034
    x = torch.load(args.load_model)
    model.load_state_dict(x['state_dict'])
           
    ''' Eval on EVP '''
    logger.print("Start testing on EVP")
    acc = val(val_loader, model, at_type)

        
        
def val(val_loader, model, at_type):
    topVideo = util.AverageMeter()

    # switch to evaluate mode
    model.eval()
    output_store_fc = []
    output_alpha    = []
    target_store = []
    index_vector = []
    
    with torch.no_grad():
        for i, (input_var, target, index) in enumerate(val_loader):
            # compute output
            target = target.to(DEVICE)
            input_var = input_var.to(DEVICE)
            ''' model & full_model'''
            f, alphas = model(input_var, phrase = 'eval')

            output_store_fc.append(f)
            output_alpha.append(alphas)
            target_store.append(target)
            index_vector.append(index)

        index_vector = torch.cat(index_vector, dim=0)  # [256] ... [256]  --->  [21570]
        index_matrix = []
        for i in range(int(max(index_vector)) + 1):
            index_matrix.append(index_vector == i)

        index_matrix = torch.stack(index_matrix, dim=0).to(DEVICE).float()  # [21570]  --->  [380, 21570]
        output_store_fc = torch.cat(output_store_fc, dim=0)  # [256,7] ... [256,7]  --->  [21570, 7]
        output_alpha    = torch.cat(output_alpha, dim=0)     # [256,1] ... [256,1]  --->  [21570, 1]
        target_store = torch.cat(target_store, dim=0).float()  # [256] ... [256]  --->  [21570]
        
        ''' keywords: mean_fc ; weight_sourcefc; sum_alpha; weightmean_sourcefc '''
        weight_sourcefc = output_store_fc.mul(output_alpha)   #[21570,512] * [21570,1] --->[21570,512]
        sum_alpha = index_matrix.mm(output_alpha) # [380,21570] * [21570,1] -> [380,1]
        weightmean_sourcefc = index_matrix.mm(weight_sourcefc).div(sum_alpha)
        target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(
            index_matrix.sum(1)).long()  # [380,21570] * [21570,1] -> [380,1] / sum([21570,1]) -> [380]

        if at_type == 'self-attention':
            pred_score = model(vm=weightmean_sourcefc, phrase='eval', AT_level='pred')
        if at_type == 'self_relation-attention':
            pred_score  = model(vectors=output_store_fc, vm=weightmean_sourcefc, alphas_from1=output_alpha, index_matrix=index_matrix, phrase='eval', AT_level='second_level')

        acc_video = util.accuracy(logger, pred_score.cpu(), target_vector.cpu(), topk=(1,), show_confusion_matrix=True, write_confusion_matrix=True)
        topVideo.update(acc_video[0], i + 1)
        logger.print(' *Acc@Video {topVideo.avg:.3f} '.format(topVideo=topVideo))

        return topVideo.avg

    
if __name__ == '__main__':
    logger = util.Logger('./log/','fan_evp')
    main()
