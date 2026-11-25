# GenAI Style Transfer Studio

A Streamlit app that applies the visual style of one image to another using neural style transfer.

## How it works

The app uses a pre-trained VGG-19 network to measure two parts of an image:

- **Content** describes the main structure and layout.
- **Style** describes textures and colors.

The app optimizes an output image to balance content from one input with style from the other. It uses Gram matrices to compare style features. The method is based on Gatys et al. (2015).

## Run locally

1. Use Python 3.9 or later.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown in the terminal. Upload a content image and a style image, adjust the settings, then select **Apply style**.

The first run downloads the pre-trained VGG-19 weights through Torchvision. A CUDA or Apple MPS device is used when available; otherwise, processing runs on the CPU.

## Settings

- **Style strength** controls how much texture and color from the style image appears in the result.
- **Content preservation** controls how much of the original structure remains.
- **Optimizer** lets you choose Adam or LBFGS.

Processing time depends on the image size, optimizer, and hardware. Images are resized to a maximum dimension of 512 pixels.

## Google Colab

`colab_setup.py` contains commands for installing dependencies and launching the app in a Colab notebook. Upload `app.py` to the notebook before running the setup script.

## Project files

```text
app.py
colab_setup.py
requirements.txt
README.md
```

## Reference

Gatys, L. A., Ecker, A. S., and Bethge, M. (2015). "A Neural Algorithm of Artistic Style." arXiv:1508.06576.
