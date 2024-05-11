import os
import time

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_PATH = "./share_model/InternVL-Chat-V1-5"
IMAGE_ROOT = "./share_data/ImageNet/ILSVRC/Data/CLS-LOC/val"
IMAGE_LIST = ["ILSVRC2012_val_00000001.JPEG", "ILSVRC2012_val_00000002.JPEG", "ILSVRC2012_val_00000003.JPEG", "ILSVRC2012_val_00000004.JPEG", "ILSVRC2012_val_00000005.JPEG", "ILSVRC2012_val_00000006.JPEG", "ILSVRC2012_val_00000007.JPEG", "ILSVRC2012_val_00000008.JPEG", "ILSVRC2012_val_00000009.JPEG", "ILSVRC2012_val_00000010.JPEG", "ILSVRC2012_val_00000011.JPEG", "ILSVRC2012_val_00000012.JPEG", "ILSVRC2012_val_00000013.JPEG", "ILSVRC2012_val_00000014.JPEG", "ILSVRC2012_val_00000015.JPEG", "ILSVRC2012_val_00000016.JPEG", "ILSVRC2012_val_00000017.JPEG", "ILSVRC2012_val_00000018.JPEG", "ILSVRC2012_val_00000019.JPEG", "ILSVRC2012_val_00000020.JPEG", "ILSVRC2012_val_00000021.JPEG", "ILSVRC2012_val_00000022.JPEG", "ILSVRC2012_val_00000023.JPEG", "ILSVRC2012_val_00000024.JPEG", "ILSVRC2012_val_00000025.JPEG",
              "ILSVRC2012_val_00000026.JPEG", "ILSVRC2012_val_00000027.JPEG", "ILSVRC2012_val_00000028.JPEG", "ILSVRC2012_val_00000029.JPEG", "ILSVRC2012_val_00000030.JPEG", "ILSVRC2012_val_00000031.JPEG", "ILSVRC2012_val_00000032.JPEG", "ILSVRC2012_val_00000033.JPEG", "ILSVRC2012_val_00000034.JPEG", "ILSVRC2012_val_00000035.JPEG", "ILSVRC2012_val_00000036.JPEG", "ILSVRC2012_val_00000037.JPEG", "ILSVRC2012_val_00000038.JPEG", "ILSVRC2012_val_00000039.JPEG", "ILSVRC2012_val_00000040.JPEG", "ILSVRC2012_val_00000041.JPEG", "ILSVRC2012_val_00000042.JPEG", "ILSVRC2012_val_00000043.JPEG", "ILSVRC2012_val_00000044.JPEG", "ILSVRC2012_val_00000045.JPEG", "ILSVRC2012_val_00000046.JPEG", "ILSVRC2012_val_00000047.JPEG", "ILSVRC2012_val_00000048.JPEG", "ILSVRC2012_val_00000049.JPEG", "ILSVRC2012_val_00000050.JPEG"]


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map='auto').eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
generation_config = dict(num_beams=1, max_new_tokens=512, do_sample=False)
question = "请详细描述这张图片."
# question = "Describe the image in detail."
start_time = time.time()
response_len = 0
img_num = len(IMAGE_LIST)

print(f"Inference {img_num} images with question {question}")
for count, img_path in enumerate(IMAGE_LIST, 1):
    img_abs_path = os.path.join(IMAGE_ROOT, img_path)
    start_loop_time = time.time()
    with torch.no_grad():
        pixel_values = load_image(
            img_abs_path, max_num=6).to(torch.bfloat16).cuda()
        response = model.chat(tokenizer, pixel_values,
                              question, generation_config)
        end_loop_time = time.time()
        loop_time = end_loop_time - start_loop_time
        response_len += len(response)
        print(
            f" == [{count}] {img_path}: took {loop_time:.2f} seconds, response length: {len(response)}")
        print(f"Question: {question} Response: {response}")

end_time = time.time()
total_time = end_time - start_time
average_time = total_time / img_num
average_response_len = response_len / img_num
print(f"Total time for {img_num} iterations: {total_time:.2f} seconds")
print(f"Average response length: {average_response_len:.2f}")
print(f"Average time per iteration: {average_time:.2f} seconds")
