"""
Google Colab setup for GenAI Style Transfer Studio
Run this in a Colab notebook to set up and run the Streamlit app
"""

# Install dependencies
!pip install -q streamlit torch torchvision Pillow

# Install streamlit-colab wrapper for Colab compatibility
!pip install -q streamlit-colab

# Run the Streamlit app
from streamlit_colab import run_streamlit

# Upload app.py to Colab first, then run:
run_streamlit("app.py")

# Alternative: If streamlit-colab doesn't work, use ngrok:
# !pip install pyngrok
# from pyngrok import ngrok
# !streamlit run app.py --server.port 8501 &
# public_url = ngrok.connect(8501)
# print(f"Public URL: {public_url}")

