from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import soundfile as sf
import torch
import numpy as np
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


class ImageAudioDataset(Dataset):
    def __init__(self, data, image_processor, audio_processor, sample_rate=16000, tokenizer=None):
        images = data['images']
        audios = data['audios']
        text = data['text']
        self.labels = data['labels']
        image_processor = image_processor
        audio_processor = audio_processor
        tokenizer = tokenizer
        if Path('./data_cache/images.pt').exists():
            self.images = torch.load('./data_cache/images.pt')
        else:
            self.images  = image_processor(images=images, return_tensors="pt").pixel_values
            # save images to pt
            torch.save(self.images, 'images.pt')
        if Path('./data_cache/audios.pt').exists():
            self.audios = torch.load('./data_cache/audios.pt')
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
            torch.save(self.audios, './data_cache/audios.pt')

        tokenizer_output = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        self.text = tokenizer_output['input_ids']
        self.attention_mask = tokenizer_output['attention_mask']
        print('===================================')
        print(len(self.text))
        print(len(self.images))
        print(len(self.audios))
        print(len(self.attention_mask))
        print(len(self.labels))
        print('===================================')
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
        if os.path.isdir(text_dir):
            text_path = os.path.join(text_dir, filename + '.txt')
            with open(text_path, 'r') as f:
                text = f.read()
        else:
            with open(text_dir, 'r') as f:
                text_dict = json.load(f)
                text = text_dict[f'{filename}.mp3']

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

        # 模态对齐投影层
        self.img_proj = torch.nn.Linear(768, 256)
        self.aud_proj = torch.nn.Linear(768, 256)
        self.txt_proj = torch.nn.Linear(768, 256)

        # self.transformer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=12)

        # 使用多层Transformer编码器（这里使用4层）
        # 跨模态Transformer
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=256, 
                nhead=8,
                dim_feedforward=1024,
                activation='gelu'
            ),
            num_layers=4
        )

        # 动态注意力池化
        self.attention_pool = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1),
            torch.nn.Softmax(dim=1)
        )

        # 分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 6),
            torch.nn.Sigmoid()
        )
        # self.linear = torch.nn.Linear(768, 6)
    def forward(self, images, audios, text, attention_mask):
        image_features = self.image_model(images).last_hidden_state
        audio_features = self.audio_model(audios).last_hidden_state
        text_features = self.text_model(text, attention_mask=attention_mask).last_hidden_state
        # print(image_features.shape,audio_features.shape,text_features.shape)
        # 模态对齐投影
        img_seq = self.img_proj(image_features) # [11,197,256]
        aud_seq = self.aud_proj(audio_features) # [11,249,256]
        txt_seq = self.txt_proj(text_features) # [11,36,256]

        # 跨模态拼接
        fused_seq = torch.cat([img_seq, aud_seq, txt_seq], dim=1)  # [11,482,256]
        # print('=======',fused_seq.shape)
        # 跨模态交互
        fused_seq = fused_seq.permute(1, 0, 2)  # Transformer需要[seq_len, batch, dim]
        # print(fused_seq.shape)
        fused_feat = self.transformer(fused_seq)  # [482,11,256]
        # print(fused_feat.shape)
        fused_feat = fused_feat.permute(1, 0, 2)  # 恢复为[11,482,256]
        # print(fused_feat.shape)
        # 动态池化
        attn_weights = self.attention_pool(fused_feat)  # [11,482,1]
        pooled_feat = torch.sum(fused_feat * attn_weights, dim=1)  # [11,256]
        
        # 分类输出
        return self.classifier(pooled_feat)  # [11,6]

def main():
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

    # TRAIN = False
    # VAL = True

    TRAIN = True
    VAL = False
    if TRAIN:
        EXP_NAME = 1
        EPOCH_NUM = 100
        ## ======================== train ============================
        os.makedirs(f'./ckpt/{EXP_NAME}', exist_ok=True)
        model = model.to(device)
        data = load_data(f'{dataset_ABAW_emi}/images/averaged', f'{dataset_ABAW_emi}/audio', f'{dataset_ABAW_emi}/text', f'{dataset_ABAW_emi}/train_split.csv')
        # data = load_data(f'{dataset_ABAW_emi}/images/averaged', f'{dataset_ABAW_emi}/audio', f'{dataset_ABAW_emi}/text_whisper_large.json', f'{dataset_ABAW_emi}/debug_train_split.csv')
        # data = load_data(f'{dataset_ABAW_emi}/images/averaged', f'{dataset_ABAW_emi}/audio', f'{dataset_ABAW_emi}/text_whisper_large.json', f'{dataset_ABAW_emi}/train_split.csv')

        dataset = ImageAudioDataset(data, image_processor, audio_processor, tokenizer=tokenizer)        # 像这样加载会不会导致dataset对象特别大，占用内存特别多
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, pin_memory=True, batch_size=32, shuffle=True)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        # 添加余弦退火调度器（周期为10个epoch）
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=EPOCH_NUM,        # 完整周期长度，此处等于EPOCH数量
            eta_min=1e-6     # 最小学习率
        )
        # 最佳模型跟踪参数
        best_loss = float('inf')
        best_epoch = 0
        for epoch in range(EPOCH_NUM):
            model.train()
            epoch_loss = 0.0
            total_samples = 0
            for images, audios, text, attention_mask, labels in dataloader:
                images = images.to(device)
                audios = audios.to(device)
                text = text.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images, audios, text, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # 累计损失
                epoch_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
            # 计算平均损失
            avg_loss = epoch_loss / total_samples
            
            # 更新学习率
            scheduler.step()

            # print(f'Epoch {epoch}, Loss: {loss.item()}')
            torch.save(model.state_dict(), f'./ckpt/{EXP_NAME}/model_{epoch}.pt')
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, f'./ckpt/{EXP_NAME}/best_model.pth')
                
                # 打印训练信息
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch} | '
                f'Loss: {avg_loss:.4f} | '
                f'LR: {current_lr:.2e} | '
                f'Best Epoch: {best_epoch} (Loss: {best_loss:.4f})')

        # 最终保存
        torch.save(model.state_dict(), f'./ckpt/{EXP_NAME}/final_model.pth')
        print(f'Training completed. Best model saved from epoch {best_epoch}')
        ## ======================== train ============================

    if VAL:
        EXP_NAME = 0
        # ======================== val ==============================
        # data = load_data(f'{dataset_ABAW_emi}/images/averaged', f'{dataset_ABAW_emi}/audio', f'{dataset_ABAW_emi}/text_whisper_large.json', f'{dataset_ABAW_emi}/test_split.csv')
        data = load_data(f'{dataset_ABAW_emi}/images/averaged', f'{dataset_ABAW_emi}/audio', f'{dataset_ABAW_emi}/text_whisper_large.json', f'{dataset_ABAW_emi}/debug_test_split.csv')
        dataset = ImageAudioDataset(data, image_processor, audio_processor, tokenizer=tokenizer)
        val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        # model.load_state_dict(torch.load('./ckpt/model_6.pt'))
        # 加载最佳权重
        checkpoint = torch.load(f'./ckpt/{EXP_NAME}/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        def get_accuracy(outputs, labels):
            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            # 计算相关系数平均值
            return np.mean([np.corrcoef(outputs[:,i], labels[:,i])[0,1] for i in range(6)])
        
        accuracies = []
        count = 0

        for images, audios,text,attention_mask, labels in val_dataloader:
            images = images.to(device)
            audios = audios.to(device)
            text = text.to(device)
            attention_mask = attention_mask.to(device)
            with torch.no_grad():
                outputs = model(images, audios, text, attention_mask)
            
            print(outputs)
            print(labels)   
            # break
            accuracy = get_accuracy(outputs, labels)
            accuracies.append(accuracy)
            count += 1
        # print(count)
        # print(f'Average accuracy: {np.mean(accuracies)}')
        # ======================== val ==============================
            
if __name__ == '__main__':
    main()
    

