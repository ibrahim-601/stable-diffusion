# Stable Diffusion demo
This is a demo project to use [Stable Diffusion](https://huggingface.co/lambdalabs/sd-image-variations-diffusers) model hosted on [banana.dev](https://www.banana.dev/).

## Testing
You can test this using any of two ways described below

### Using API 
To test the model using API with different SDKs follow the instructions from [banana.dev](https://docs.banana.dev/banana-docs/core-concepts/sdks).

### Using provided script
You can also test this model using provided [clinet.py](./client.py) script. To use this script follow below instructions.
1. Install dependencies by executing below command in a terminal.
```bash
pip install Pillow
pip install gradio
pip install banana_dev
```
2. Open [clinet.py](./client.py) in a text editor and set your API key and MODEL key from [banana.dev](https://www.banana.dev/) and save the file
3. Run the script by executing below command in a terminal.
```bash
python client.py
```
3. Previos command will show a link in the terminal. Open the link in a browser.
4. Use the GUI to select image, change parameters (Guidence scale, Steps, etc.) and press the `Submit` button.
5. After sometime output will be shown at right side of the screen.