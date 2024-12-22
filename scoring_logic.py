import easyocr
import numpy as np
import cv2
from PIL import ImageColor, Image
from scipy.spatial import KDTree
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ImageScorer:
    def __init__(self, image_path, brand_palette, generated_prompt=None):
        """
        Initialize the ImageScorer with the image path, brand palette, and an optional prompt.
        
        Args:
            image_path (str): Path to the image.
            brand_palette (list): List of hex colors in the brand palette.
            generated_prompt (str, optional): The generated prompt text used for text similarity comparison.
        """
        self.image_path = image_path
        self.brand_palette = [
            ImageColor.getrgb(color.strip()) for color in brand_palette
        ]  # Strip whitespace and convert hex to RGB
        self.image = self._load_image()
        self.generated_prompt = generated_prompt

    def _load_image(self):
        """
        Load the image from the given path.

        Returns:
            PIL.Image.Image: Loaded image.
        """
        return Image.open(self.image_path)

    def calculate_palette_contrast(self):
        """
        Calculate the color contrast using the brand palette.

        Returns:
            float: Color contrast value.
        """
        def luminance(color):
            return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]

        if not self.brand_palette:
            return 0.0  # Handle case where brand_palette is empty

        max_luminance = max(luminance(color) for color in self.brand_palette)
        min_luminance = min(luminance(color) for color in self.brand_palette)

        return (max_luminance + 0.05) / (min_luminance + 0.05)

    def extract_palette_details_in_image(self):
        """
        Extract color details from the image based on the brand palette.

        Returns:
            dict: Percentage contribution of each color in the palette.
        """
        image_rgb = self.image.convert("RGB")
        pixels = np.array(image_rgb).reshape(-1, 3)  # Flatten the image to a list of pixels

        # Create a KDTree for fast nearest-neighbor lookup
        tree = KDTree(self.brand_palette)

        # Find the nearest palette color for each pixel
        _, indices = tree.query(pixels)

        # Count occurrences of each palette color
        unique, counts = np.unique(indices, return_counts=True)
        total_pixels = len(pixels)

        # Calculate percentage contribution
        color_contribution = {
            tuple(self.brand_palette[i]): round((count / total_pixels) * 100, 2)
            for i, count in zip(unique, counts)
        }

        return color_contribution

    def extract_luminance_details_in_image(self):
        """
        Analyze the luminance of the image and judge exposure levels.

        Returns:
            dict: Luminance histogram, exposure judgment, and percentages.
        """
        image_cv = cv2.imread(self.image_path)

        # Convert the image to RGB and calculate luminance
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        luminance = 0.2126 * image_rgb[:, :, 0] + 0.7152 * image_rgb[:, :, 1] + 0.0722 * image_rgb[:, :, 2]

        # Calculate the histogram
        histogram, _ = np.histogram(luminance, bins=256, range=(0, 255))

        # Calculate exposure percentages
        total_pixels = luminance.size
        underexposed_percentage = np.sum(histogram[:85]) / total_pixels * 100  # Pixels with luminance [0, 85)
        overexposed_percentage = np.sum(histogram[170:]) / total_pixels * 100  # Pixels with luminance [170, 255)
        normal_percentage = 100 - (underexposed_percentage + overexposed_percentage)

        # Determine exposure judgment
        if underexposed_percentage > 50:
            exposure = "Underexposed"
        elif overexposed_percentage > 50:
            exposure = "Overexposed"
        else:
            exposure = "Normal Exposure"

        return {
            "underexposed_percentage": round(underexposed_percentage, 2),
            "normal_percentage": round(normal_percentage, 2),
            "overexposed_percentage": round(overexposed_percentage, 2),
            "exposure_judgment": exposure
        }

    def extract_text_from_image(self):
        """
        Extract text from the image using EasyOCR.

        Returns:
            str: Extracted text from the image.
        """
        reader = easyocr.Reader(['en'])  # Initialize the EasyOCR reader with English language
        results = reader.readtext(self.image_path)

        # Extract the text from the results
        extracted_text = ' '.join([result[1] for result in results])
        return extracted_text

    def compute_text_similarity(self, extracted_text):
        """
        Compute similarity between the generated prompt and extracted text using cosine similarity.

        Args:
            extracted_text (str): The text extracted from the image.

        Returns:
            float: Similarity score between 0 and 1.
        """
        if not self.generated_prompt:
            return 0.0  # Return 0 similarity if no generated prompt

        # Initialize the TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')

        # Combine the prompt and extracted text into a list
        documents = [self.generated_prompt, extracted_text]

        # Fit and transform the documents into TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Compute cosine similarity between the two TF-IDF vectors
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

        # Return the similarity score (a value between 0 and 1)
        return similarity_matrix[0][0]

    def calculate_scores(self):
        """
        Perform all calculations and return base scores for the image.

        Returns:
            dict: Scores and details about the image.
        """
        # Perform all calculations and return the scores
        scores = {
            "palette_contrast": self.calculate_palette_contrast(),
            "palette_details": self.extract_palette_details_in_image(),
            "luminance_details": self.extract_luminance_details_in_image(),
            "extracted_text": self.extract_text_from_image()
        }
        return scores

    def calculate_enhanced_scores(self):
        """
        Perform all calculations and return enhanced scores for the image,
        including text similarity score.

        Returns:
            dict: Enhanced scores and details about the image.
        """
        # Calculate base scores
        scores = self.calculate_scores()

        # Compute text similarity score
        extracted_text = scores.get("extracted_text", "")
        text_similarity_score = self.compute_text_similarity(extracted_text)

        # Add the text similarity score to the scores dictionary
        scores["text_similarity_score"] = text_similarity_score

        return scores
