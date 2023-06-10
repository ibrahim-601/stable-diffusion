import base64
from io import BytesIO

from PIL import Image
import gradio as gr
import banana_dev as banana

API_KEY = "your_api_key"
MODEL_KEY = "you_model_key"

# function to save base64 string as image
def base64_to_image(base64_images):
    images = []
    for base64_image in base64_images:
        im = Image.open(BytesIO(base64.b64decode(base64_image)))
        images.append(im)
    return images

# function convert image to base64 string
def image_to_base64(input_image):
    input_image = Image.fromarray(input_image)
    buffered = BytesIO()
    input_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = img_str.decode('utf-8')
    return img_str

# main function to send request and receive response
def main(input_im, scale=3.0, n_samples=4, steps=25, seed=0):
    img_str = image_to_base64(input_im)
    inp = {
        "base64_input": img_str,
        "output_images_number": n_samples,
        "guidance_scale": scale,
        "steps": steps,
        "seed": seed
    }
    api_outputs = banana.run(API_KEY, MODEL_KEY, inp)
    images = base64_to_image(api_outputs["modelOutputs"][0]["base64_images"])
    return images

# Description text shown in gradio app
description = \
"""
__Now using Image Variations v2!__

Generate variations on an input image using a fine-tuned version of Stable Diffision.
Trained by [Justin Pinkney](https://www.justinpinkney.com) ([@Buntworthy](https://twitter.com/Buntworthy)) at [Lambda](https://lambdalabs.com/)

This version has been ported to 🤗 Diffusers library, see more details on how to use this version in the [Lambda Diffusers repo](https://github.com/LambdaLabsML/lambda-diffusers).
For the original training code see [this repo](https://github.com/justinpinkney/stable-diffusion).

![](https://raw.githubusercontent.com/justinpinkney/stable-diffusion/main/assets/im-vars-thin.jpg)

"""

# article text shown in gradio app
article = \
"""
## How does this work?

The normal Stable Diffusion model is trained to be conditioned on text input. This version has had the original text encoder (from CLIP) removed, and replaced with
the CLIP _image_ encoder instead. So instead of generating images based a text input, images are generated to match CLIP's embedding of the image.
This creates images which have the same rough style and content, but different details, in particular the composition is generally quite different.
This is a totally different approach to the img2img script of the original Stable Diffusion and gives very different results.

The model was fine tuned on the [LAION aethetics v2 6+ dataset](https://laion.ai/blog/laion-aesthetics/) to accept the new conditioning.
Training was done on 8xA100 GPUs on [Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud).
More details are on the [model card](https://huggingface.co/lambdalabs/sd-image-variations-diffusers).
"""

# inputs for gradio
inputs = [
    gr.Image(),
    gr.Slider(0, 25, value=3, step=1, label="Guidance scale"),
    gr.Slider(1, 4, value=1, step=1, label="Number images"),
    gr.Slider(5, 50, value=25, step=5, label="Steps"),
    gr.Number(0, label="Seed", precision=0)
]

# gradio app section for displaying output
output = gr.Gallery(label="Generated variations")
output.style(grid=2)

# exaple images to load
examples = [
    ["examples/vermeer.jpg", 3, 1, 25, 0],
    ["examples/matisse.jpg", 3, 1, 25, 0],
]

# create gradio app interface
sd_app = gr.Interface(
    fn=main,
    title="Stable Diffusion Image Variations",
    description=description,
    article=article,
    inputs=inputs,
    outputs=output,
    examples=examples,
    )

# launch gradio app
sd_app.launch()
