from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import soundfile as sf
import torch
import numpy as np
import csv
import os
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path


from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2Model
from transformers import BertTokenizer,AutoTokenizer
from transformers import BertModel
import pandas as pd
from tqdm import tqdm
import json


class ImageAudioDataset_val(Dataset):
    def __init__(self, data, image_processor, audio_processor, sample_rate=16000, tokenizer=None):
        self.filenames = data['filenames']
        images = data['images']
        audios = data['audios']
        text = data['text']
        self.labels = data['labels']
        image_processor = image_processor
        audio_processor = audio_processor
        tokenizer = tokenizer
        if Path('./data_cache/images_val.pt').exists():
            self.images = torch.load('./data_cache/images_val.pt')
        else:
            self.images  = image_processor(images=images, return_tensors="pt").pixel_values
            # save images to pt
            torch.save(self.images, './data_cache/images_val.pt')
        if Path('./data_cache/audios_val.pt').exists():
            self.audios = torch.load('./data_cache/audios_val.pt')
        else:
            self.audios = []
            for audio in tqdm(audios, total=len(audios)):
                audio = audio[:sample_rate*5]
                audio = audio_processor(audio, return_tensors="pt", sampling_rate=sample_rate).input_values
                # pad audio to 5s
                if audio.shape[1] < sample_rate*5:
                    audio = torch.cat((audio, torch.zeros((1, sample_rate*5 - audio.shape[1]))), dim=1)
                self.audios.append(audio)
            # save audios to pt
            torch.save(self.audios, './data_cache/audios_val.pt')

        tokenizer_output = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        self.text = tokenizer_output['input_ids']
        self.attention_mask = tokenizer_output['attention_mask']
        print('===================================')
        print('filenames: ',len(self.filenames))
        print('text: ',len(self.text))
        print('images: ',len(self.images))
        print('audios: ',len(self.audios))
        print('attention_mask: ',len(self.attention_mask))
        print('labels: ',len(self.labels))
        print('===================================')
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.filenames[idx], self.images[idx], self.audios[idx].squeeze(0), self.text[idx], self.attention_mask[idx], torch.tensor(self.labels[idx])
    
def load_data(image_dir, audio_dir,text_dir, label_dir):
    data = {
        'filenames': [],
        'images': [],
        'audios': [],
        'text': [],
        'labels': []
    }
    labels_df = pd.read_csv(label_dir, names=['Filename','Admiration','Amusement','Determination','Empathic Pain','Excitement','Joy'], header=0)
    for index, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        # 格式化filename为五位数
        filename = str(int(row['Filename']))
        filename = filename.zfill(5)
        
        # 使用 with 确保图像文件被关闭
        image_path = os.path.join(image_dir, filename + '_average.png')
        with Image.open(image_path) as img:
            image = img.convert("RGB").copy()
        # 使用 with 读取音频文件，利用 SoundFile 上下文管理器
        audio_path = os.path.join(audio_dir, filename + '.mp3')
        with sf.SoundFile(audio_path) as f:
            audio = f.read(dtype='float32')
        if os.path.isdir(text_dir):
            text_path = os.path.join(text_dir, filename + '.txt')
            with open(text_path, 'r') as f:
                text = f.read()
        else:
            with open(text_dir, 'r') as f:
                text_dict = json.load(f)
                text = text_dict[f'{filename}.mp3']
        data['filenames'].append(filename)
        data['text'].append(text)
        data['images'].append(image)
        data['audios'].append(audio)
        data['labels'].append(row[1:].values.tolist())
    return data


def main():
    # from main_train1 import ImageAudioModel
    # EXP_NAME = 'train1'

    from main_train1_position_emb import ImageAudioModel
    EXP_NAME = 'main_train1_position_emb'

    # from main_triple_loss import ImageAudioModel
    # EXP_NAME = 'triple_loss_weight0.2'

    # from main_triple_loss_position_emb import ImageAudioModel
    # EXP_NAME = 'train_triple_loss_position_emb'

    image_processor = AutoImageProcessor.from_pretrained("/home/data2/zls/code/ckpt/google/vit-base-patch16-224-in21k")
    audio_processor = Wav2Vec2Processor.from_pretrained("/home/data2/zls/code/ckpt/facebook/wav2vec2-base-960h")
    
    image_model = AutoModel.from_pretrained('/home/data2/zls/code/ckpt/google/vit-base-patch16-224-in21k')
    audio_model = Wav2Vec2Model.from_pretrained("/home/data2/zls/code/ckpt/facebook/wav2vec2-base-960h")
    
    # tokenizer = BertTokenizer.from_pretrained("/home/data2/zls/code/ckpt/google-bert/bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("/home/data2/zls/code/ckpt/google-bert/bert-base-uncased")
    text_model = BertModel.from_pretrained('/home/data2/zls/code/ckpt/google-bert/bert-base-uncased')
    
    dataset_ABAW_emi = '/home/data2/zls/code/ABAW/emi/dataset'
    
    model = ImageAudioModel(image_model, audio_model, text_model)

    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")  # CPU设备对象

    # ======================== val ==============================
    # data = load_data(f'{dataset_ABAW_emi}/images/averaged', f'{dataset_ABAW_emi}/audio', f'{dataset_ABAW_emi}/text', f'{dataset_ABAW_emi}/valid_split.csv')
    # data = load_data(f'{dataset_ABAW_emi}/images/averaged', f'{dataset_ABAW_emi}/audio', f'{dataset_ABAW_emi}/text', f'{dataset_ABAW_emi}/debug_valid_split.csv')
    # data = load_data(f'{dataset_ABAW_emi}/images/averaged', f'{dataset_ABAW_emi}/audio', f'{dataset_ABAW_emi}/text_whisper_large.json', f'{dataset_ABAW_emi}/valid_split.csv')
    data = load_data(f'{dataset_ABAW_emi}/images/averaged', f'{dataset_ABAW_emi}/audio', f'{dataset_ABAW_emi}/text_whisper_large.json', f'{dataset_ABAW_emi}/debug_valid_split.csv')
    dataset = ImageAudioDataset_val(data, image_processor, audio_processor, tokenizer=tokenizer)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    # model.load_state_dict(torch.load('./ckpt/model_6.pt'))
    # 加载最佳权重
    checkpoint = torch.load(f'./ckpt/{EXP_NAME}/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    def get_accuracy(outputs, labels):
        # outputs = outputs.cpu().detach().numpy()
        # labels = labels.cpu().detach().numpy()
        # 计算相关系数平均值
        # return np.mean([np.corrcoef(outputs[:,i], labels[:,i])[0,1] for i in range(6)])

        # 计算相关系数平均值（带异常值处理）
        corr_values = []
        for i in range(6):
            x = outputs[:, i]
            y = labels[:, i]
            
            # 检查数据有效性
            if np.std(x) == 0 and np.std(y) == 0:
                # 两者都是常数，视为完全相关
                corr = 1.0
            elif np.std(x) == 0 or np.std(y) == 0:
                # 任一变量是常数，相关系数未定义，视为0
                corr = 0.0
            else:
                corr = np.corrcoef(x, y)[0, 1]
                
            corr_values.append(corr)
        
        return np.nanmean(corr_values)  # 使用nanmean忽略可能的NaN
    
    accuracies = []
    output_list = []
    label_list = []
    filename_list = []
    os.makedirs('./outputs', exist_ok=True)
    outputs_save_path = f'./outputs/submission_{EXP_NAME}.csv'
    count = 0

    for filename, images, audios, text, attention_mask, labels in tqdm(val_dataloader, total=len(val_dataloader)):
        images = images.to(device)
        audios = audios.to(device)
        text = text.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(images, audios, text, attention_mask)
            # outputs = outputs[0]        # triple loss 
            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
        filename_list.extend(filename)
        output_list.extend(outputs)
        label_list.extend(labels)
        print(outputs)
        print(labels)   
        accuracy = get_accuracy(outputs, labels)
        accuracies.append(accuracy)
        print(f'Accuracy: {accuracy}')
        print("====================")
        count += 1


    with open(outputs_save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        header = ["Filename", "Admiration", "Amusement", "Determination", "Empathic Pain", "Excitement", "Joy"]
        writer.writerow(header)
        # 写入数据行
        for i, (filename, label, output) in tqdm(enumerate(zip(filename_list, label_list, output_list))):
            data_row = [filename] + output.tolist()
            writer.writerow(data_row)

    # 计算总体平均的准确率
    print(count)
    print(accuracies)
    print(f'Average accuracy: {np.mean(accuracies)}')
    with open(f'./outputs/val_{EXP_NAME}.txt', 'w') as f:
        f.write(f'Average accuracy: {np.mean(accuracies)}')
    # ======================== val ==============================
        
if __name__ == '__main__':
    main()
    

