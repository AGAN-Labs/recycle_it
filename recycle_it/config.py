from pathlib import Path

image_endpoint = "http://localhost:8000/images/"
module_dir = Path(__file__).parent.parent
test_image_path = module_dir.joinpath('data/test_data/20210601_152952.jpg')