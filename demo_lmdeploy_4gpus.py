from lmdeploy import TurbomindEngineConfig, pipeline
from lmdeploy.vl import load_image

pipe = pipeline("./share_model/InternVL-Chat-V1-5",
                backend_config=TurbomindEngineConfig(tp=4, cache_max_entry_count=0.2))
image = load_image("./example.jpg")
response = pipe(("Describe the image in detail.", image))
print(response)
