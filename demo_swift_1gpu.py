import torch
from swift.llm import (ModelType, get_default_template_type,
                       get_model_tokenizer, get_template, inference)

model_type = ModelType.internvl_chat_v1_5_int8
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

# for GPUs that do not support flash attention
# torch.bfloat16 but use_flash_attn = False
model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                       model_kwargs={'device_map': 'auto'},
                                       model_id_or_path="./share_model/InternVL-Chat-V1-5-Int8",
                                       use_flash_attn=False)

model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
question = "Describe the image in detail."
response, history = inference(model, template, question, images=[
                              './example.jpg'])  # chat with image
print(response)
