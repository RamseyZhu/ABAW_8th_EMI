from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import soundfile as sf
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from pathlib import Path

from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2Model
from transformers import BertTokenizer
from transformers import BertModel
import pandas as pd
from tqdm import tqdm


class ImageAudioDataset(Dataset):
    def __init__(self, data, image_processor, audio_processor, sample_rate=16000, tokenizer=None):
        images = data['images']
        audios = data['audios']
        text = data['text']
        self.labels = data['labels']
        image_processor = image_processor
        audio_processor = audio_processor
        tokenizer = tokenizer
        if Path('images.pt').exists():
            self.images = torch.load('images.pt')
        else:
            self.images  = image_processor(images=images, return_tensors="pt").pixel_values
        # save images to pt
        torch.save(self.images, 'images.pt')
        if Path('audios.pt').exists():
            self.audios = torch.load('audios.pt')
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
        torch.save(self.audios, 'audios.pt')
        tokenizer_output = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        self.text = tokenizer_output['input_ids']
        self.attention_mask = tokenizer_output['attention_mask']
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.audios[idx].squeeze(0), self.text[idx], self.attention_mask[idx], torch.tensor(self.labels[idx])
    
def load_data(image_dir, audio_dir,text_dir, label_dir):
    data = {
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
        text_path = os.path.join(text_dir, filename + '.txt')
        with open(text_path, 'r') as f:
            text = f.read()
        data['text'].append(text)
        data['images'].append(image)
        data['audios'].append(audio)
        data['labels'].append(row[1:].values.tolist())
    return data


class ImageAudioModel(torch.nn.Module):
    def __init__(self, image_model, audio_model, text_model):
        super(ImageAudioModel, self).__init__()
        self.image_model = image_model
        self.audio_model = audio_model
        self.text_model = text_model
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=12)
        self.linear = torch.nn.Linear(768, 6)
    def forward(self, images, audios, text, attention_mask):
        image_features = self.image_model(images).last_hidden_state
        text_features = self.text_model(text, attention_mask=attention_mask).last_hidden_state
        image_features = image_features[:,0,:].unsqueeze(1)  # [batch_size, 1, 768]
        text_features = text_features[:,0,:].unsqueeze(1)    # [batch_size, 1, 768]
        
        return image_features, text_features

def main():
    image_processor = AutoImageProcessor.from_pretrained("/data/maihn/Bert/vit-base-patch16-224-in21k")
    audio_processor = Wav2Vec2Processor.from_pretrained("/data/maihn/Bert/wav2vec2-base-960h")
    
    image_model = AutoModel.from_pretrained('/data/maihn/Bert/vit-base-patch16-224-in21k')
    audio_model = Wav2Vec2Model.from_pretrained("/data/maihn/Bert/wav2vec2-base-960h")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_model = BertModel.from_pretrained('bert-base-uncased')
    
    data = load_data('ABAW/images/averaged', 'ABAW/audio', 'ABAW/text', 'ABAW/train_split.csv')
    
    model = ImageAudioModel(image_model, audio_model, text_model)
    model = model.to('cuda:4')
    dataset = ImageAudioDataset(data, image_processor, audio_processor, tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    def CLIPLoss(image_features, text_features):
        return 1 - torch.cosine_similarity(image_features, text_features, dim=-1).mean()
    criterion = CLIPLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for epoch in range(10):
        for images, audios, text, attention_mask, labels in dataloader:
            images = images.to('cuda:4')
            audios = audios.to('cuda:4')
            text = text.to('cuda:4')
            attention_mask = attention_mask.to('cuda:4')
            labels = labels.to('cuda:4')
            optimizer.zero_grad()
            image_features, text_features = model(images, audios, text, attention_mask)
            loss = criterion(image_features, text_features)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), f'model_{epoch}.pt')

    # val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    # model.load_state_dict(torch.load('model_3.pt'))

    # def get_accuracy(outputs, labels):
    #     outputs = outputs.cpu().detach().numpy()
    #     labels = labels.cpu().detach().numpy()
    #     # 计算相关系数平均值
    #     return np.mean([np.corrcoef(outputs[:,i], labels[:,i])[0,1] for i in range(6)])
    
    # accuracies = []
    # count = 0

    # for images, audios, labels in val_dataloader:
    #     images = images.cuda()
    #     audios = audios.cuda()
    #     labels = labels.cuda()
    #     outputs = model(images, audios)
    #     print(outputs)
    #     print(labels)   
    #     break
    #     accuracy = get_accuracy(outputs, labels)
    #     accuracies.append(accuracy)
    #     count += 1

    # print(f'Average accuracy: {np.mean(accuracies)}')

            
if __name__ == '__main__':
    main()
    

