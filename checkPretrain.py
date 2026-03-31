

import torch
old_model = torch.load("training_output/cuoi_cham_finetune_pretrain_September-16-2025_09+25AM-f5a66b3/checkpoint_1223026.pth")
print("Embedding shape ignored:", old_model['model']['text_encoder.emb.weight'].shape)


# Load pretrained model
model_path = "/home/anhht/.local/share/tts/tts_models--en--ljspeech--vits/model_file.pth"
checkpoint = torch.load(model_path, map_location='cpu')

print("Embedding shape pretrained model: ",checkpoint['model']['text_encoder.emb.weight'].shape) 
