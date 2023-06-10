import base64
from io import BytesIO
from potassium import Potassium, Request, Response
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
import torch
from torchvision import transforms

app = Potassium("stable-diffusion-app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    model = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers",
        )
    model = model.to(device)
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
    device = 0 if torch.cuda.is_available() else -1
    inp = tform(im).to(device).unsqueeze(0)
    return inp

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    base64_image = request.json.get("base64_input")
    processed_image = pre_process_image(base64_image)
    model = context.get("model")
    outputs = model(processed_image, guidance_scale=3)
    output = outputs["images"][0]
    if(outputs["nsfw_content_detected"][0]):
        output = Image.open(r"unsafe.png")
    buffered = BytesIO()
    output.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return Response(
        json = {"output": img_str}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()