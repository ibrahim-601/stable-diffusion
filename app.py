import base64
from io import BytesIO
from potassium import Potassium, Request, Response
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
import torch
from torchvision import transforms

app = Potassium("stable-diffusion-app")
DEVICE = 0 if torch.cuda.is_available() else -1
# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    model = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers",
        )
    model = model.to(DEVICE)
    context = {
        "model": model
    }

    return context

# function to process image
def pre_process_image(base64_image):
    im = Image.open(BytesIO(base64.b64decode(base64_image)))
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]),
    ])
    inp = tform(im).to(DEVICE).unsqueeze(0)
    return inp

# convert images to base64 string
def image_to_base64(images_list):
    output = []
    for i, image in enumerate(images_list):
        if(images_list["nsfw_content_detected"][i]):
            image = Image.open(r"unsafe.png")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode('utf-8')
        output.append(img_str)
    return output

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    base64_image = request.json.get("base64_input")
    n_samples = request.json.get("output_images_number")
    scale = request.json.get("guidance_scale")
    steps = request.json.get("steps")
    seed = request.json.get("seed")
    processed_image = pre_process_image(base64_image)
    model = context.get("model")
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    outputs = model(processed_image.tile(n_samples, 1, 1, 1), guidance_scale=scale, num_inference_steps=steps, generator=generator)
    image_strings = image_to_base64(outputs["images"])    
    return Response(
        json = {"base64_images": image_strings}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()