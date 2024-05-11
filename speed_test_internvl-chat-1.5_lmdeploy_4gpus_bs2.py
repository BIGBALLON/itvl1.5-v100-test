import os
import time

from lmdeploy import TurbomindEngineConfig, pipeline
from PIL import Image

MODEL_PATH = "./share_model/InternVL-Chat-V1-5"
IMAGE_ROOT = "./share_data/ImageNet/ILSVRC/Data/CLS-LOC/val"
IMAGE_LIST = ["ILSVRC2012_val_00000001.JPEG", "ILSVRC2012_val_00000002.JPEG", "ILSVRC2012_val_00000003.JPEG", "ILSVRC2012_val_00000004.JPEG", "ILSVRC2012_val_00000005.JPEG", "ILSVRC2012_val_00000006.JPEG", "ILSVRC2012_val_00000007.JPEG", "ILSVRC2012_val_00000008.JPEG", "ILSVRC2012_val_00000009.JPEG", "ILSVRC2012_val_00000010.JPEG", "ILSVRC2012_val_00000011.JPEG", "ILSVRC2012_val_00000012.JPEG", "ILSVRC2012_val_00000013.JPEG", "ILSVRC2012_val_00000014.JPEG", "ILSVRC2012_val_00000015.JPEG", "ILSVRC2012_val_00000016.JPEG", "ILSVRC2012_val_00000017.JPEG", "ILSVRC2012_val_00000018.JPEG", "ILSVRC2012_val_00000019.JPEG", "ILSVRC2012_val_00000020.JPEG", "ILSVRC2012_val_00000021.JPEG", "ILSVRC2012_val_00000022.JPEG", "ILSVRC2012_val_00000023.JPEG", "ILSVRC2012_val_00000024.JPEG", "ILSVRC2012_val_00000025.JPEG",
              "ILSVRC2012_val_00000026.JPEG", "ILSVRC2012_val_00000027.JPEG", "ILSVRC2012_val_00000028.JPEG", "ILSVRC2012_val_00000029.JPEG", "ILSVRC2012_val_00000030.JPEG", "ILSVRC2012_val_00000031.JPEG", "ILSVRC2012_val_00000032.JPEG", "ILSVRC2012_val_00000033.JPEG", "ILSVRC2012_val_00000034.JPEG", "ILSVRC2012_val_00000035.JPEG", "ILSVRC2012_val_00000036.JPEG", "ILSVRC2012_val_00000037.JPEG", "ILSVRC2012_val_00000038.JPEG", "ILSVRC2012_val_00000039.JPEG", "ILSVRC2012_val_00000040.JPEG", "ILSVRC2012_val_00000041.JPEG", "ILSVRC2012_val_00000042.JPEG", "ILSVRC2012_val_00000043.JPEG", "ILSVRC2012_val_00000044.JPEG", "ILSVRC2012_val_00000045.JPEG", "ILSVRC2012_val_00000046.JPEG", "ILSVRC2012_val_00000047.JPEG", "ILSVRC2012_val_00000048.JPEG", "ILSVRC2012_val_00000049.JPEG", "ILSVRC2012_val_00000050.JPEG"]
BATCH_SIZE = 2


def load_image(img_path):
    return Image.open(img_path).convert("RGB")


def process_prompts(prompts, pipe):
    start_loop_time = time.time()
    response = pipe(prompts)
    end_loop_time = time.time()
    loop_time = end_loop_time - start_loop_time
    response_len = sum(len(res.text) for res in response)
    return loop_time, response_len, response


if __name__ == '__main__':
    pipe = pipeline(MODEL_PATH, backend_config=TurbomindEngineConfig(
        tp=4, cache_max_entry_count=0.2))
    question = "请详细描述这张图片."
    # question = "Describe the image in detail."
    start_time = time.time()
    total_response_len = 0
    img_num = len(IMAGE_LIST)
    prompts = []
    print(f"Inference {len(IMAGE_LIST)} images with question {question}")

    for count, img_path in enumerate(IMAGE_LIST, 1):
        prompts.append((question, load_image(
            os.path.join(IMAGE_ROOT, img_path))))
        if len(prompts) == BATCH_SIZE:
            loop_time, response_len, response = process_prompts(prompts, pipe)
            total_response_len += response_len
            print(f" == [{count}] {img_path}: took {loop_time:.2f} seconds")
            for res in response:
                print(f"Question: {question} Response: {res.text}")
            prompts = []
    if len(prompts):
        loop_time, response_len, response = process_prompts(prompts, pipe)
        total_response_len += response_len
        print(f" == [{count}] {img_path}: took {loop_time:.2f} seconds")
        for res in response:
            print(f"Question: {question} Response: {res.text}")

    total_time = time.time() - start_time
    average_time = total_time / len(IMAGE_LIST)
    average_response_len = total_response_len / img_num
    print(
        f"Total time for {len(IMAGE_LIST)} iterations: {total_time:.2f} seconds")
    print(f"Average time per iteration: {average_time:.2f} seconds")
    print(f"Average response length: {average_response_len:.2f}")
