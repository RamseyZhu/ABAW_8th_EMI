import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
from torchmetrics.regression import PearsonCorrCoef

def evaluate(config, model, eval_dataloader):
    with torch.no_grad():
        preds, labels, filenames = predict(config, model, eval_dataloader)
        r = PearsonCorrCoef(num_outputs=6)
        r = r(preds, labels)
        r = r.mean()
    return r.cpu().numpy(), preds, filenames


def predict(train_config, model, dataloader):
    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    r = PearsonCorrCoef(num_outputs=6).cuda()
    preds = []
    labels = []
    filenames = []
    with torch.no_grad():

        # for wav2vec, vit, label, filename in bar:
        for wav2vec, label, filename in bar:

            with autocast():

                wav2vec = wav2vec.to(train_config.device)
                # vit = vit.to(train_config.device)
                label = label.to(train_config.device)
                # pred = model(wav2vec, vit)
                pred = model(wav2vec)

            # save features in fp32 for sim calculation
            labels.append(label.detach().cpu())
            preds.append(pred.to(torch.float32).detach().cpu())
            filenames.extend(filename)
            r.update(pred, label)
            bar.set_postfix(ordered_dict={'corr': f'{r.compute().mean().cpu().numpy():.4f}'})
        # keep Features on GPU
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

    if train_config.verbose:
        bar.close()

    return preds, labels, filenames
