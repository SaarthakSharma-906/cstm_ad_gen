import streamlit as st
import requests

API_ENDPOINT = "http://127.0.0.1:8000/generate_ad/"

st.set_page_config(page_title="Ad Creative Generator", page_icon="💡", layout="wide")

st.title("💡 Ad Creative Generator & Scorer")

# User Inputs Section with Enhanced Styling
with st.container():
    st.header("Enter Your Ad Creative Details")
    
    # Use columns to organize input fields
    col1, col2 = st.columns(2)
    with col1:
        product_name = st.text_input("Product Name", "GlowWell Skin Serum")
        tagline = st.text_input("Tagline", "Radiance Redefined.")
        colors = st.text_area("Brand Palette (Comma-separated)", "#FFC107, #212121, #FFFFFF")
    with col2:
        cta_text = st.text_input("Call to Action", "Shop Now")
        logo_url = st.text_input("Logo URL", "https://example.com/logo.png")
        product_image_url = st.text_input("Product Image URL", "https://example.com/product.png")

# Submit Button
if st.button("Generate Creative"):
    # Prepare input data
    input_data = {
        "creative_details": {
            "product_name": product_name,
            "tagline": tagline,
            "brand_palette": colors.split(","),
            "dimensions": {"width": 1080, "height": 1080},
            "cta_text": cta_text,
            "logo_url": logo_url,
            "product_image_url": product_image_url
        },
        "scoring_criteria": {
            "background_foreground_separation": 20,
            "brand_guideline_adherence": 20,
            "creativity_visual_appeal": 20,
            "product_focus": 20,
            "call_to_action": 20
        }
    }

    # Make API call
    response = requests.post(API_ENDPOINT, json=input_data)

    if response.status_code == 200:
        result = response.json()

        # Display the generated creative as a clickable link
        creative_url = result["creative_url"]
        st.markdown(f"**[Click here to view the generated image]({creative_url})**", unsafe_allow_html=True)

        # Display scoring breakdown with improved styling
        st.markdown("""
        <style>
        .scoring-card {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            background-color: #f9f9f9;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 30px;
        }
        .scoring-card h4 {
            font-size: 20px;
            color: #4CAF50;
        }
        .scoring-card .score {
            font-size: 22px;
            font-weight: bold;
            color: #333;
        }
        .scoring-card p {
            margin: 8px 0;
            color: #666;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="scoring-card">', unsafe_allow_html=True)
        st.markdown("<h4>Scoring Breakdown</h4>", unsafe_allow_html=True)
        
        # Display scores dynamically
        scoring = result["scoring"]
        for category, score in scoring.items():
            st.markdown(f"<p><strong>{category.replace('_', ' ').title()}:</strong> <span class='score'>{score}%</span></p>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error(f"Error: {response.status_code}")
