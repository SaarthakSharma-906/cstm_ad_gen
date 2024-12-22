from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.stable_diffusion_api import generate_image_from_hf
from api.scoring_logic import ImageScorer  # Import the updated ImageScorer
from utils.prompt_generator import create_prompt
import boto3
import os
import uuid
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FastAPI app instance
app = FastAPI()

# Input model for validation
class CreativeDetails(BaseModel):
    product_name: str
    tagline: str
    brand_palette: list
    dimensions: dict
    cta_text: str
    logo_url: str
    product_image_url: str

class RequestPayload(BaseModel):
    creative_details: CreativeDetails
    scoring_criteria: dict

# Initialize S3 client
s3_client = boto3.client("s3")

def upload_to_s3(file_path, bucket_name, s3_key):
    """
    Upload a file to S3 and return the uploaded file's S3 key.

    Args:
        file_path (str): Local file path to upload.
        bucket_name (str): S3 bucket name.
        s3_key (str): Destination path/key in the S3 bucket.

    Returns:
        str: Uploaded file's S3 key.
    """
    try:
        s3_client.upload_file(
            file_path, bucket_name, s3_key,
            ExtraArgs={"ContentType": "image/png"}
        )
        return s3_key
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading to S3: {str(e)}")

def generate_presigned_url(bucket_name, object_name, expiration=3600):
    """
    Generate a presigned URL for the object in the S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket.
        object_name (str): Path to the object in the bucket.
        expiration (int): Time in seconds for the presigned URL to remain valid.

    Returns:
        str: Presigned URL as a string.
    """
    try:
        response = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_name},
            ExpiresIn=expiration
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating presigned URL: {str(e)}")

def compute_text_similarity(prompt, extracted_text):
    """
    Compute similarity between the generated prompt and extracted text.
    Using a lightweight TF-IDF vectorization and cosine similarity.
    
    Args:
        prompt (str): The generated prompt.
        extracted_text (str): The extracted text from the image.

    Returns:
        float: Similarity score between 0 and 1.
    """
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Combine the prompt and extracted text into a list
    documents = [prompt, extracted_text]

    # Fit and transform the documents into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute cosine similarity between the two TF-IDF vectors
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    # Return the similarity score (a value between 0 and 1)
    return similarity_matrix[0][0]

@app.post("/generate_ad/")
async def generate_ad(payload: RequestPayload):
    try:
        # Extract creative details
        details = payload.creative_details
        prompt = create_prompt(
            brand_title=details.product_name,
            tagline=details.tagline,
            cta=details.cta_text,
            additional_description=" ".join(details.brand_palette)
        )
        
        # Generate image using Stable Diffusion
        image_data = generate_image_from_hf(prompt)  # This returns a dict containing the image path
        image_path = image_data.get('data')  # Ensure 'data' contains the correct path

        # Ensure that the image is in the correct format (if it's .webp, convert to .png)
        if isinstance(image_path, str) and image_path.endswith('.webp'):
            # Convert .webp to .png
            local_file = f"assets/{uuid.uuid4().hex}.png"
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            img = Image.open(image_path)
            img.save(local_file, 'PNG')  # Save as PNG
        else:
            # If the image is already a compatible format, save it directly
            local_file = f"assets/{uuid.uuid4().hex}.png"
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            with open(local_file, "wb") as f:
                f.write(image_data["data"])  # Writing image data to file

        # Initialize the ImageScorer to compute scores
        image_scorer = ImageScorer(image_path=local_file, brand_palette=details.brand_palette)
        scoring = image_scorer.calculate_enhanced_scores()

        # Compute text similarity score
        extracted_text = scoring.get("extracted_text", "")
        similarity_score = compute_text_similarity(prompt, extracted_text)

        # Add the text similarity score to the scoring data
        scoring["text_similarity_score"] = similarity_score

        # Debugging: Log the scoring result to see what is missing
        print(f"Scoring result: {scoring}")

        # Check if 'palette_contrast' exists, and handle the case if not
        if 'palette_contrast' not in scoring:
            raise HTTPException(status_code=500, detail="Missing 'palette_contrast' in scoring data")

        # Flatten or serialize any lists that might be causing serialization issues
        scoring_serialized = {
            "palette_contrast": scoring.get("palette_contrast"),
            "palette_details": {str(key): value for key, value in scoring.get("palette_details", {}).items()},
            "luminance_details": scoring.get("luminance_details", {}),
            "extracted_text": extracted_text,
            "background_foreground_separation": scoring.get("background_foreground_separation"),
            "brand_guideline_adherence": scoring.get("brand_guideline_adherence"),
            "creativity_visual_appeal": scoring.get("creativity_visual_appeal"),
            "product_focus": scoring.get("product_focus"),
            "call_to_action": scoring.get("call_to_action"),
            "audience_relevance": scoring.get("audience_relevance"),
            "text_similarity_score": similarity_score  # Include text similarity score
        }

        # Upload to S3 bucket
        bucket_name = "your-s3-bucket-name"  # Replace with your actual bucket name
        s3_key = f"generated_images/{uuid.uuid4().hex}.png"
        upload_to_s3(local_file, bucket_name, s3_key)

        # Generate a presigned URL to access the uploaded image
        presigned_url = generate_presigned_url(bucket_name, s3_key)

        # Return the URL and scoring as the response
        return {"creative_url": presigned_url, "scoring": scoring_serialized}

    except Exception as e:
        # Log the exception for debugging
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
