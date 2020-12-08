
import argparse
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os

def load_save_model(model_name, des_path):
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    print(f'开始加载模型{model_name}')
    #加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #加载config
    config = AutoConfig.from_pretrained(model_name)
    #加载模型
    model = AutoModel.from_pretrained(model_name)
    #保存模型
    tokenizer.save_pretrained(des_path)
    config.save_pretrained(des_path)
    model.save_pretrained(des_path)
    #删除不需要的配置，防止不兼容
    os.remove(os.path.join(des_path,'tokenizer_config.json'))
    os.remove(os.path.join(des_path,'special_tokens_map.json'))
    print(f'模型 {model_name} 已保存到 {des_path} ')

if __name__ == '__main__':
    load_save_model(model_name='bert-base-chinese', des_path='pytorch_version/prev_trained_model/bert-base')