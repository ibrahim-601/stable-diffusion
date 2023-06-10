# This file runs during container build time to get model weights built into the container

# In this example: A StableDiffusion Image Variation model
from diffusers import StableDiffusionImageVariationPipeline

def download_model():
    # do a dry run of loading the StableDiffusion Image Variation model, which will download weights
    pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
    )
if __name__ == "__main__":
    download_model()