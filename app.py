"""Interactive image stylization with neural style transfer."""

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
from io import BytesIO

# ==========================================
# CONFIGURATION
# ==========================================
MAX_IMAGE_SIZE = 512
# VGG-19 layer indices (using ReLU layers after convolutions)
# Style layers: Multiple layers at different scales to capture texture patterns
STYLE_LAYERS = [1, 6, 11, 20, 29]  # After conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
CONTENT_LAYER = 21  # After conv4_2 - captures high-level structure/shape
NUM_STEPS = 250  # Reduced for faster generation (was 500, taking 15-20 min)
OPTIMIZER_TYPE = "adam"  # "adam" for faster, "lbfgs" for slightly better quality
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="GenAI Style Transfer Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    h1, h2, h3 { color: #8ab4f8; }
    .stText {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# VGG MODEL LOADING
# ==========================================
@st.cache_resource
def load_vgg_model():
    """Load pre-trained VGG-19 model"""
    vgg = models.vgg19(weights='DEFAULT').features.eval()
    # Freeze parameters
    for param in vgg.parameters():
        param.requires_grad_(False)
    vgg = vgg.to(DEVICE)
    return vgg

# ==========================================
# IMAGE PROCESSING
# ==========================================
def load_image(image_file, max_size=MAX_IMAGE_SIZE):
    """Load and preprocess image"""
    if isinstance(image_file, BytesIO):
        image = Image.open(image_file).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    
    # Resize while maintaining aspect ratio
    size = image.size
    if max(size) > max_size:
        scale = max_size / max(size)
        new_size = (int(size[0] * scale), int(size[1] * scale))
        image = image.resize(new_size, Image.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(DEVICE), image

def tensor_to_image(tensor):
    """Convert tensor back to PIL Image"""
    tensor = tensor.cpu().clone()
    tensor = tensor.squeeze(0)
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    
    tensor = tensor.clamp(0, 1)
    transform = transforms.ToPILImage()
    image = transform(tensor)
    return image

# ==========================================
# FEATURE EXTRACTION
# ==========================================
def get_features(model, x, style_layers, content_layer):
    """Extract features from specific VGG layers"""
    style_features = {}
    content_feature = None
    
    for i, layer in enumerate(model):
        x = layer(x)
        if i in style_layers:
            style_features[i] = x
        if i == content_layer:
            content_feature = x
    
    return style_features, content_feature

def gram_matrix(tensor):
    """Calculate Gram matrix for style features"""
    batch_size, n_channels, height, width = tensor.size()
    features = tensor.view(batch_size, n_channels, height * width)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (n_channels * height * width)

# ==========================================
# STYLE TRANSFER
# ==========================================
def neural_style_transfer(content_img, style_img, vgg_model, 
                         content_weight, style_weight, num_steps, 
                         progress_bar, status_text, use_adam=True):
    """
    Perform neural style transfer
    
    This combines visual structure from one image with textures and colors from another:
    - Content: The main objects and layout from the content image
    - Style: The textures and colors from the style image
    
    The algorithm learns to synthesize new patterns by optimizing pixel values through
    backpropagation, making it a true generative AI approach.
    """
    
    # Extract features
    status_text.text("Extracting Features from VGG-19...")
    
    # Get content and style features
    with torch.no_grad():
        style_feats, _ = get_features(vgg_model, style_img, STYLE_LAYERS, CONTENT_LAYER)
        _, content_feat_target = get_features(vgg_model, content_img, STYLE_LAYERS, CONTENT_LAYER)
    
    content_features = {CONTENT_LAYER: content_feat_target}
    style_features = style_feats
    
    # Calculate style Gram matrices (capture texture patterns)
    status_text.text("Computing Style Gram Matrices...")
    style_grams = {layer: gram_matrix(style_features[layer]) 
                   for layer in style_features}
    
    # Initialize generated image (start with content to preserve structure)
    generated_img = content_img.clone().requires_grad_(True)
    
    # Choose optimizer: Adam is faster, LBFGS is slightly better quality
    if use_adam:
        optimizer = optim.Adam([generated_img], lr=0.02)
    else:
        optimizer = optim.LBFGS([generated_img], lr=0.1, max_iter=20)
    
    status_text.text("Applying the selected style...")
    
    if use_adam:
        # Adam optimizer: straightforward loop
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Extract features from generated image
            gen_style_feats, gen_content_feat = get_features(vgg_model, generated_img, STYLE_LAYERS, CONTENT_LAYER)
            
            # Content loss preserves the structure of the original image
            content_loss = nn.MSELoss()(gen_content_feat, content_features[CONTENT_LAYER])
            
            # Style loss matches textures and colors from the style image
            style_loss = 0
            for layer in style_grams:
                gen_gram = gram_matrix(gen_style_feats[layer])
                style_loss += nn.MSELoss()(gen_gram, style_grams[layer])
            style_loss /= len(style_grams)
            
            # Total loss: weighted combination
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Clamp to valid image range (will be denormalized later)
            with torch.no_grad():
                # Clamp in normalized space (rough bounds)
                generated_img.data.clamp_(-2.5, 2.5)
            
            # Update progress
            if step % 25 == 0 or step == num_steps - 1:
                progress = (step + 1) / num_steps
                progress_bar.progress(progress)
                status_text.text(f"Generating... Step {step + 1}/{num_steps} (Loss: {total_loss.item():.2f})")
    else:
        # LBFGS optimizer: uses closure function
        step_count = [0]
        
        def closure():
            nonlocal step_count
            step_count[0] += 1
            
            optimizer.zero_grad()
            
            # Extract features from generated image
            gen_style_feats, gen_content_feat = get_features(vgg_model, generated_img, STYLE_LAYERS, CONTENT_LAYER)
            
            # Content loss
            content_loss = nn.MSELoss()(gen_content_feat, content_features[CONTENT_LAYER])
            
            # Style loss
            style_loss = 0
            for layer in style_grams:
                gen_gram = gram_matrix(gen_style_feats[layer])
                style_loss += nn.MSELoss()(gen_gram, style_grams[layer])
            style_loss /= len(style_grams)
            
            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            total_loss.backward()
            
            # Update progress
            if step_count[0] % 20 == 0:
                progress = min(step_count[0] / num_steps, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Rendering... Step {step_count[0]}/{num_steps}")
            
            return total_loss
        
        # Optimization loop for LBFGS
        for i in range(num_steps // 20):
            optimizer.step(closure)
            if step_count[0] >= num_steps:
                break
    
    progress_bar.progress(1.0)
    status_text.text("Style transfer complete!")
    
    return generated_img

# ==========================================
# STREAMLIT UI
# ==========================================
def main():
    st.title("🎨 GenAI Style Transfer Studio")
    st.markdown("Apply the visual style of one image to another with neural style transfer.")
    
    # Sidebar
    with st.sidebar:
        st.header("Style Transfer Settings")
        
        st.subheader("Input Images")
        content_file = st.file_uploader(
            "Content image",
            type=['png', 'jpg', 'jpeg'],
            help="Choose the image whose composition you want to preserve."
        )
        
        style_file = st.file_uploader(
            "Style image",
            type=['png', 'jpg', 'jpeg'],
            help="Choose an image whose colors and textures you want to apply."
        )
        
        st.divider()
        
        st.subheader("🎛️ Control Parameters")
        style_weight = st.slider(
            "Style strength",
            min_value=1000.0,
            max_value=1000000.0,
            value=30000.0,
            step=5000.0,
            help="Higher values emphasize textures and colors from the style image."
        )
        
        content_weight = st.slider(
            "Content preservation",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            help="Higher values preserve more of the content image."
        )
        
        st.divider()
        
        optimizer_choice = st.radio(
            "Optimizer",
            ["Adam (Faster ~3-5 min)", "LBFGS (Better Quality ~8-12 min)"],
            index=0,
            help="Adam is faster, LBFGS produces slightly better results"
        )
        
        st.divider()
        
        generate_button = st.button(
            "Apply style",
            type="primary",
            use_container_width=True
        )
        
        st.divider()
        
        st.subheader("📊 System Status")
        status_text = st.empty()
        status_text.text("Ready. Add two images to begin.")
        
        progress_bar = st.progress(0)
    
    # Main content area
    if content_file and style_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Content image")
            content_pil = Image.open(content_file).convert('RGB')
            st.image(content_pil, use_container_width=True)
        
        with col2:
            st.subheader("Style image")
            style_pil = Image.open(style_file).convert('RGB')
            st.image(style_pil, use_container_width=True)
        
        if generate_button:
            try:
                # Reset file pointers if needed
                if hasattr(content_file, 'seek'):
                    content_file.seek(0)
                if hasattr(style_file, 'seek'):
                    style_file.seek(0)
                
                # Load images
                status_text.text("Loading Images...")
                content_tensor, _ = load_image(content_file)
                style_tensor, _ = load_image(style_file)
                
                # Load model
                status_text.text("Loading VGG-19 Model...")
                vgg_model = load_vgg_model()
                
                # Perform neural style transfer
                use_adam = optimizer_choice.startswith("Adam")
                
                with st.spinner("Applying neural style transfer..."):
                    result_tensor = neural_style_transfer(
                        content_tensor, 
                        style_tensor,
                        vgg_model,
                        content_weight,
                        style_weight,
                        NUM_STEPS,
                        progress_bar,
                        status_text,
                        use_adam=use_adam
                    )
                
                # Convert to PIL Image
                result_image = tensor_to_image(result_tensor)
                
                # Display result
                st.divider()
                st.subheader("Result")
                
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.markdown("**Content image**")
                    st.image(content_pil, use_container_width=True)
                
                with result_col2:
                    st.markdown("**Style image**")
                    st.image(style_pil, use_container_width=True)
                
                with result_col3:
                    st.markdown("**Stylized result**")
                    st.image(result_image, use_container_width=True)
                
                # Download button
                st.divider()
                buf = BytesIO()
                result_image.save(buf, format="PNG")
                st.download_button(
                    label="Download result",
                    data=buf.getvalue(),
                    file_name="style_transfer_result.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                status_text.text("Style transfer complete.")
                
            except Exception as e:
                st.error(f"Style transfer failed: {str(e)}")
                status_text.text(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    elif generate_button:
        st.warning("Please upload both a content image and a style image.")
    
    else:
        st.info("Upload a content image and a style image using the sidebar.")
        
        # Show example/demo info
        with st.expander("📖 How to Use"):
            st.markdown("""
            ### Instructions:
            1. **Choose a content image**: This provides the composition and structure of the result.
            2. **Choose a style image**: This provides the textures and colors.
            3. **Adjust Parameters**: 
               - **Style strength** controls how strongly the style image affects the result.
               - **Content preservation** controls how much structure is kept from the content image.
            4. Click **Apply style** and wait for the result.
            5. Download the result as a PNG image.
            
            ### Technical Details:
            - **Method**: Neural style transfer, based on Gatys et al. (2015)
            - **Feature extractor**: VGG-19 pre-trained on ImageNet
            - **Optimizers**: Adam or LBFGS
            - **Device**: Uses CUDA or Apple MPS when available, otherwise CPU
            - **Performance**: 
              - CPU: ~5-8 minutes (Adam) or ~10-15 minutes (LBFGS)
              - GPU: ~1-2 minutes (Adam) or ~3-5 minutes (LBFGS)
              - **For faster results, use Google Colab with GPU runtime**
            """)
        
        with st.expander("How neural style transfer works"):
            st.markdown("""
            Neural style transfer separates visual features into content and style.
            A pre-trained VGG-19 network measures the content structure and the style
            textures. The app then adjusts the input image to balance those two objectives.
            Style features are compared using Gram matrices.

            The method is based on Gatys et al., "A Neural Algorithm of Artistic Style"
            (2015).
            """)
        
        with st.expander("🚀 Google Colab Setup (Recommended for Faster Results)"):
            st.markdown("""
            ### To Run on Google Colab for GPU Speed:
            
            1. **Create a new Colab notebook**
            2. **Upload `app.py` to Colab**
            3. **Install dependencies**:
               ```python
               !pip install streamlit streamlit-colab
               !pip install torch torchvision Pillow
               ```
            4. **Run Streamlit**:
               ```python
               from streamlit_colab import run_streamlit
               run_streamlit("app.py")
               ```
            
            **Benefits**:
            - Free GPU access (T4 or better)
            - Much faster generation (~1-2 minutes vs 5-15 minutes)
            - No local setup required
            """)
    
    # Footer
    st.divider()
    st.caption("GenAI Style Transfer Studio | PyTorch, Streamlit, and neural style transfer")

if __name__ == "__main__":
    main()

