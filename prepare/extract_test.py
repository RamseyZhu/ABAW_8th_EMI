import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import soundfile as sf


device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_id = "/home/data2/zls/code/ckpt/openai/whisper-large-v3-turbo"
# model_id = "/home/data2/zls/code/ckpt/whisper-large-v3"
# model_id = "openai/whisper-large-v3-turbo"
# model_id = "/home/data2/zls/code/ckpt/openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)


# audio_dir = Path('ABAW/audio')
audio_dir = Path('/home/data2/zls/code/ABAW/emi/dataset/test_data/audio')
audio_files = list(audio_dir.glob('*.mp3'))
# output_dir = Path('ABAW/text')
output_dir = Path('/home/data2/zls/code/ABAW/emi/dataset/test_data/text')

output_dir.mkdir(exist_ok=True)
output_dir_list = [f.stem for f in output_dir.glob("*.txt")]

for audio_file in tqdm(audio_files):
    if audio_file.stem in output_dir_list:
        continue
    try:
        audio_path = str(audio_file)
        audio, sample_rate = sf.read(audio_path)
        inputs = processor(audio, return_tensors="pt", sampling_rate=sample_rate).input_features
        inputs = inputs.to(device)
        with torch.no_grad():
            generated_ids = model.generate(input_features=inputs)
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        output_file = output_dir / f"{audio_file.stem}.txt"
        with open(output_file, 'w') as f:
            f.write(result)
    except Exception as e:
        print(f"无法处理 {audio_file}: {e}")
            
