import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import openai
import self
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from flask.cli import load_dotenv
from scipy.signal import find_peaks

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Handwriting Analysis System")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HandwritingFeatureExtractor:
    """Class for extracting features from handwriting images"""

    def __init__(self):
        # Initialize feature extraction parameters
        self.pressure_thresholds = {
            'light': 120,
            'medium': 180,
            'heavy': 255
        }
        self.size_thresholds = {
            'small': 20,
            'medium': 50,
            'large': float('inf')
        }

    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract all features from the image"""
        try:
            features = {
                'baseline': self.analyze_baseline(image),
                'slant': self.analyze_slant(image),
                'pressure': self.analyze_pressure(image),
                'size': self.analyze_size(image),
                'spacing': self.analyze_spacing(image),
                'zones': self.analyze_zones(image),
                't_crosses': self.analyze_t_crosses(image),
                'i_dots': self.analyze_i_dots(image),
                'letter_connections': self.analyze_letter_connections(image),
                'signature': self.analyze_signature(image),
                't_bar': self.analyze_t_bar(image)
            }
            return features
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            raise

    def analyze_baseline(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze baseline characteristics"""
        try:
            # Get horizontal projection
            projection = np.sum(image, axis=1)
            peaks, properties = find_peaks(projection, height=np.mean(projection))

            if len(peaks) < 2:
                return {'direction': 'unknown', 'consistency': 0, 'angle': 0}

            # Calculate line angles
            angles = []
            for i in range(len(peaks) - 1):
                y_diff = peaks[i + 1] - peaks[i]
                x_diff = 100  # Fixed horizontal distance
                angle = math.degrees(math.atan2(y_diff, x_diff))
                angles.append(angle)

            avg_angle = np.mean(angles)
            angle_std = np.std(angles)
            consistency = 1 - (angle_std / 90)  # Normalize standard deviation

            # Determine baseline direction
            if abs(avg_angle) < 5:
                direction = 'horizontal'
            elif avg_angle > 5:
                direction = 'ascending'
            else:
                direction = 'descending'

            return {
                'direction': direction,
                'consistency': float(consistency),
                'angle': float(avg_angle),
                'variation': angle_std
            }
        except Exception as e:
            logger.error(f"Error in baseline analysis: {str(e)}")
            return {'direction': 'unknown', 'consistency': 0, 'angle': 0}

    def analyze_spacing(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze writing spacing characteristics"""
        try:
            # Horizontal projection to find word separation
            horizontal_projection = np.sum(image, axis=1)
            word_separation_peaks, _ = find_peaks(horizontal_projection, distance=20)

            # Vertical projection to find letter spacing
            vertical_projection = np.sum(image, axis=0)
            letter_separation_peaks, _ = find_peaks(vertical_projection, distance=10)

            # Calculate spacing metrics
            if len(word_separation_peaks) > 1 and len(letter_separation_peaks) > 1:
                avg_word_spacing = np.mean(np.diff(word_separation_peaks))
                avg_letter_spacing = np.mean(np.diff(letter_separation_peaks))

                # Normalize spacing
                max_height, max_width = image.shape
                word_spacing_ratio = avg_word_spacing / max_height
                letter_spacing_ratio = avg_letter_spacing / max_width

                # Categorize spacing
                def categorize_spacing(ratio):
                    if ratio < 0.1:
                        return 'tight'
                    elif ratio < 0.2:
                        return 'medium'
                    else:
                        return 'wide'

                return {
                    'word_spacing': float(word_spacing_ratio),
                    'letter_spacing': float(letter_spacing_ratio),
                    'word_spacing_category': categorize_spacing(word_spacing_ratio),
                    'letter_spacing_category': categorize_spacing(letter_spacing_ratio)
                }

            return {
                'word_spacing': 0,
                'letter_spacing': 0,
                'word_spacing_category': 'unknown',
                'letter_spacing_category': 'unknown'
            }

        except Exception as e:
            logger.error(f"Error in spacing analysis: {str(e)}")
            return {
                'word_spacing': 0,
                'letter_spacing': 0,
                'word_spacing_category': 'unknown',
                'letter_spacing_category': 'unknown'
            }

    def analyze_zones(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze writing zones (upper, middle, lower)"""
        try:
            height = image.shape[0]

            # Divide image into three zones
            upper_zone = image[:height // 3, :]
            middle_zone = image[height // 3:2 * height // 3, :]
            lower_zone = image[2 * height // 3:, :]

            # Calculate zone content intensity
            def calculate_zone_intensity(zone):
                return np.mean(zone) / 255 if zone.size > 0 else 0

            upper_intensity = calculate_zone_intensity(upper_zone)
            middle_intensity = calculate_zone_intensity(middle_zone)
            lower_intensity = calculate_zone_intensity(lower_zone)

            # Determine dominant zone
            zone_intensities = {
                'upper': upper_intensity,
                'middle': middle_intensity,
                'lower': lower_intensity
            }
            dominant_zone = max(zone_intensities, key=zone_intensities.get)

            # Determine zone distribution
            total_intensity = upper_intensity + middle_intensity + lower_intensity
            zone_distribution = {
                'upper': upper_intensity / total_intensity if total_intensity > 0 else 0,
                'middle': middle_intensity / total_intensity if total_intensity > 0 else 0,
                'lower': lower_intensity / total_intensity if total_intensity > 0 else 0
            }

            return {
                'dominant_zone': dominant_zone,
                'distribution': {k: float(v) for k, v in zone_distribution.items()},
                'zone_intensities': {k: float(v) for k, v in zone_intensities.items()}
            }

        except Exception as e:
            logger.error(f"Error in zone analysis: {str(e)}")
            return {
                'dominant_zone': 'unknown',
                'distribution': {'upper': 0, 'middle': 0, 'lower': 0},
                'zone_intensities': {'upper': 0, 'middle': 0, 'lower': 0}
            }

    def analyze_slant(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze writing slant"""
        try:
            # Apply edge detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)

            # Use probabilistic Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                    threshold=50,
                                    minLineLength=30,
                                    maxLineGap=10)

            if lines is None or len(lines) == 0:
                return {'angle': 0, 'consistency': 0, 'variation': 'unknown'}

            # Calculate angles
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:  # Avoid division by zero
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    if -90 <= angle <= 90:  # Filter valid angles
                        angles.append(angle)

            if not angles:
                return {'angle': 0, 'consistency': 0, 'variation': 'unknown'}

            avg_angle = np.mean(angles)
            consistency = 1 - (np.std(angles) / 90)

            # Determine slant category
            if abs(avg_angle) < 5:
                variation = 'vertical'
            elif avg_angle > 5:
                variation = 'right'
            else:
                variation = 'left'

            return {
                'angle': float(avg_angle),
                'consistency': float(consistency),
                'variation': variation
            }
        except Exception as e:
            logger.error(f"Error in slant analysis: {str(e)}")
            return {'angle': 0, 'consistency': 0, 'variation': 'unknown'}

    def analyze_pressure(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze writing pressure"""
        try:
            # Calculate intensity statistics
            intensity = np.mean(image)
            std_dev = np.std(image)

            # Determine pressure level
            if intensity < self.pressure_thresholds['light']:
                level = 'light'
            elif intensity < self.pressure_thresholds['medium']:
                level = 'medium'
            else:
                level = 'heavy'

            # Calculate consistency
            consistency = 1 - (std_dev / 255)

            # Create pressure map
            pressure_map = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)

            return {
                'level': level,
                'consistency': float(consistency),
                'average_intensity': float(intensity / 255),
                'variation': float(std_dev / 255),
                'pressure_points': self.find_pressure_points(image)
            }
        except Exception as e:
            logger.error(f"Error in pressure analysis: {str(e)}")
            return {'level': 'unknown', 'consistency': 0}

    def find_pressure_points(self, image: np.ndarray) -> List[Dict[str, int]]:
        """Find points of high pressure in writing"""
        try:
            # Threshold for high pressure points
            threshold = np.percentile(image, 90)

            # Find coordinates of high pressure points
            y_coords, x_coords = np.where(image > threshold)

            # Group nearby points
            points = []
            for y, x in zip(y_coords, x_coords):
                points.append({'x': int(x), 'y': int(y)})

            return points[:10]  # Return top 10 points
        except Exception as e:
            logger.error(f"Error finding pressure points: {str(e)}")
            return []

    def analyze_size(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze writing size"""
        try:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

            if num_labels < 2:  # Less than 2 means only background
                return {
                    'overall': 'unknown',
                    'consistency': 0,
                    'average_height': 0
                }

            # Get heights of all components (excluding background)
            heights = stats[1:, cv2.CC_STAT_HEIGHT]

            # Calculate average height
            avg_height = np.mean(heights)

            # Determine size category
            if avg_height < 20:
                size = 'small'
            elif avg_height > 50:
                size = 'large'
            else:
                size = 'medium'

            # Calculate consistency (1 - normalized standard deviation)
            consistency = 1 - (np.std(heights) / avg_height if avg_height > 0 else 0)

            return {
                'overall': size,
                'consistency': float(consistency),
                'average_height': float(avg_height)
            }
        except Exception as e:
            logger.error(f"Error in size analysis: {str(e)}")
            return {
                'overall': 'unknown',
                'consistency': 0,
                'average_height': 0
            }

    def analyze_t_crosses(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze t-bar characteristics"""
        try:
            return {
                'height_ratio': 0.5,
                'length_ratio': 1.0,
                'angle': 0,
                'pressure': 'medium'
            }
        except Exception as e:
            logger.error(f"Error in t-cross analysis: {str(e)}")
            return {
                'height_ratio': 0,
                'length_ratio': 0,
                'angle': 0,
                'pressure': 'unknown'
            }

    def analyze_i_dots(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze i-dot characteristics"""
        try:
            return {
                'position': 'centered',
                'pressure': 'medium',
                'shape': 'round'
            }
        except Exception as e:
            logger.error(f"Error in i-dot analysis: {str(e)}")
            return {
                'position': 'unknown',
                'pressure': 'unknown',
                'shape': 'unknown'
            }

    def analyze_letter_connections(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze letter connections"""
        try:
            return {
                'connection_type': 'garland',
                'consistency': 0.8,
                'spacing': 'medium'
            }
        except Exception as e:
            logger.error(f"Error in letter connection analysis: {str(e)}")
            return {
                'connection_type': 'unknown',
                'consistency': 0,
                'spacing': 'unknown'
            }

    def analyze_signature(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze signature characteristics such as confidence, ego, and privacy.
        Args:
            image (np.ndarray): Input binary image of the signature.
        Returns:
            Dict[str, Any]: Signature analysis results.
        """
        try:
            # Calculate size and proportions
            height, width = image.shape
            size_ratio = height / width

            # Detect legibility using edge intensity
            edges = cv2.Canny(image, 50, 150)
            edge_density = np.sum(edges) / (height * width)

            # Determine traits
            if size_ratio > 0.3 and edge_density > 0.2:
                confidence = "High"
            elif size_ratio < 0.2:
                confidence = "Low"
            else:
                confidence = "Medium"

            ego = "Strong" if width > 200 else "Balanced"
            privacy = "Open" if edge_density > 0.25 else "Private"

            return {
                "confidence": confidence,
                "ego": ego,
                "privacy": privacy,
                "size_ratio": round(size_ratio, 2),
                "edge_density": round(edge_density, 2)
            }
        except Exception as e:
            logger.error(f"Error in signature analysis: {str(e)}")
            return {"confidence": "unknown", "ego": "unknown", "privacy": "unknown"}

    def analyze_t_bar(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze T-bar height and alignment for leadership traits.
        Args:
            image (np.ndarray): Binary image of handwriting.
        Returns:
            Dict[str, Any]: T-bar height ratio and alignment.
        """
        try:
            # Detect horizontal lines
            edges = cv2.Canny(image, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=20, maxLineGap=5)

            if not lines is None:
                bar_positions = [y for _, y, _, _ in lines[:, 0] if y > 0]
                avg_position = np.mean(bar_positions) if bar_positions else 0

                height = image.shape[0]
                ratio = avg_position / height

                position_category = "High" if ratio < 0.3 else "Medium" if ratio < 0.6 else "Low"
                return {"t_bar_position": position_category, "height_ratio": round(ratio, 2)}
            return {"t_bar_position": "unknown", "height_ratio": 0}

        except Exception as e:
            logger.error(f"Error analyzing T-bar: {str(e)}")
            return {"t_bar_position": "unknown", "height_ratio": 0}


class HandwritingAnalyzer:
    def __init__(self):
        self.feature_extractor = HandwritingFeatureExtractor()
        self.setup_resources()

    def setup_resources(self):
        """Initialize necessary resources and directories"""
        try:
            # Step 1: Create required directories
            directories = ['models', 'data/results', 'data/uploads', 'logs']
            for dir_path in directories:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logging.info(f"Directory {dir_path} created or already exists.")

            # Step 2: Retrieve API Key from Environment Variables
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your environment.")

            # Step 3: Initialize OpenAI API
            openai.api_key = self.openai_api_key
            logging.info("OpenAI API key successfully loaded.")

            # Step 4: Load ML models
            self.load_models()

        except ValueError as e:
            logging.error(f"Environment setup error: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in setup_resources: {e}")
            raise

    def load_models(self):
        """Load or initialize ML models"""
        try:
            self.models = {}
            model_types = ['slant', 'pressure', 'baseline', 'size']

            for model_type in model_types:
                model_path = f'models/{model_type}_model.h5'
                if os.path.exists(model_path):
                    self.models[model_type] = tf.keras.models.load_model(model_path)
                else:
                    logger.info(f"Creating new model for {model_type}")
                    self.models[model_type] = self.create_new_model(model_type)
                    self.models[model_type].save(model_path)

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def create_new_model(self, model_type: str) -> tf.keras.Model:
        """Create a new ML model"""
        try:
            # Define model architecture
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(64, 64, 1)),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(self.get_output_size(model_type), activation='softmax')
            ])

            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            return model

        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise

    def get_output_size(self, model_type: str) -> int:
        """Get the output size for different model types"""
        sizes = {
            'slant': 3,  # left, vertical, right
            'pressure': 3,  # light, medium, heavy
            'baseline': 3,  # ascending, horizontal, descending
            'size': 3  # small, medium, large
        }
        return sizes.get(model_type, 3)

    async def analyze_image(self, file: UploadFile) -> Dict[str, Any]:
        """Main analysis pipeline"""
        try:
            # Read and preprocess image
            contents = await file.read()
            image = self.preprocess_image(contents)

            # Extract features
            features = self.feature_extractor.extract_features(image)

            # Get AI analysis
            analysis = await self.get_gpt_analysis(features)

            # Prepare result
            result = {
                'features': features,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }

            # Save result
            self.save_result(result)

            return result

        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def preprocess_image(self, contents: bytes) -> np.ndarray:
        """Preprocess image for analysis"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # Denoise
            denoised = cv2.fastNlMeansDenoising(thresh)

            # Resize if needed
            if max(denoised.shape) > 1000:
                scale = 1000 / max(denoised.shape)
                new_size = tuple(int(dim * scale) for dim in denoised.shape[:2])
                denoised = cv2.resize(denoised, new_size[::-1])

            return denoised

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    async def get_gpt_analysis(self, features: Dict[str, Any]) -> str:
        """Get analysis from GPT"""
        try:
            from openai import AsyncOpenAI

            # Initialize AsyncOpenAI client
            client = AsyncOpenAI(api_key=self.openai_api_key)

            prompt = self.create_analysis_prompt(features)

            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert graphologist with deep knowledge of handwriting analysis and psychology."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1500,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in GPT analysis: {str(e)}")
            return "Analysis unavailable due to error"

    def create_analysis_prompt(self, features: Dict[str, Any]) -> str:
        """Create narrative-style prompt for GPT analysis with holistic coaching insights"""
        return f""" As an expert graphologist, psychologist, and multidisciplinary coach, analyze the following 
        handwriting and signature features. Provide a comprehensive, engaging, and actionable summary that integrates 
        **graphology insights** with perspectives from a **health coach, mental coach, career coach, and mentor**.

        ### **Objectives of the Analysis**: 1. Use handwriting features to describe the writer's personality, 
        emotional tendencies, cognitive patterns, and social behavior. 2. Provide tailored recommendations in four 
        key areas: - **Health and Wellness**: Physical and mental well-being (e.g., stress levels, 
        energy management). - **Emotional and Mental Coaching**: Resilience, focus, emotional balance, 
        and stress-handling abilities. - **Career Guidance**: Professional strengths, leadership potential, 
        creativity, and areas for growth. - **Personal Development and Mentorship**: Communication skills, 
        relationships, confidence-building, and self-discipline.

        ### **Handwriting Features**:
        **Baseline Analysis**:
        - Direction: {features['baseline']['direction']}
        - Consistency: {features['baseline']['consistency']:.2f}
        - Angle: {features['baseline']['angle']:.2f} degrees

        **Slant Analysis**:
        - Variation: {features['slant']['variation']}
        - Angle: {features['slant']['angle']:.2f} degrees
        - Consistency: {features['slant']['consistency']:.2f}

        **Pressure Analysis**:
        - Level: {features['pressure']['level']}
        - Consistency: {features['pressure']['consistency']:.2f}
        - Average Intensity: {features['pressure'].get('average_intensity', 0):.2f}

        **Size Analysis**:
        - Overall Size: {features['size']['overall']}
        - Consistency: {features['size']['consistency']:.2f}
        - Average Height: {features['size'].get('average_height', 0):.2f}

        **Spacing Analysis**:
        - Word Spacing: {features['spacing']['word_spacing_category']}
        - Letter Spacing: {features['spacing']['letter_spacing_category']}

        **Zone Analysis**:
        - Dominant Zone: {features['zones']['dominant_zone']}
        - Zone Distribution: {features['zones']['distribution']}

        **Signature Analysis**:
        - Confidence: {features['signature']['confidence']}
        - Ego: {features['signature']['ego']}
        - Privacy: {features['signature']['privacy']}
        - Size Ratio: {features['signature']['size_ratio']:.2f}

        **T-Bar Analysis**:
        - Position: {features['t_bar']['t_bar_position']}
        - Height Ratio: {features['t_bar']['height_ratio']:.2f}

        ### **Guidelines for Analysis**:
        - **Personality Insights**:
          Describe the writer’s confidence, ambition, creativity, and emotional state.
        - **Health and Wellness Coaching**:
          Based on handwriting pressure, slant, and zones, identify energy levels, stress markers, and suggest health habits for physical and mental well-being.
        - **Emotional and Mental Coaching**:
          Provide recommendations on stress-handling strategies, focus, emotional resilience, and mindfulness practices.
        - **Career Coaching**:
          Highlight career strengths (e.g., analytical thinking, leadership skills, creativity), suitable roles, and areas for professional growth.
        - **Mentorship and Development**:
          Suggest ways to improve communication, build stronger relationships, enhance confidence, and work on self-discipline for personal development.

        ### **Example Output**:
        "The writer exhibits high confidence and leadership tendencies, shown by their firm baseline and bold signature. However, inconsistent pressure indicates fluctuating energy levels, suggesting a need for better stress management and work-life balance.  
        From a **health perspective**, incorporating mindfulness practices and physical activity would help stabilize energy.  
        Their dominant middle zone and medium word spacing show practical thinking and respect for boundaries, making them well-suited for analytical or managerial roles.  
        As a **career coach**, I recommend exploring leadership positions where decision-making and empathy are valued.  
        To build resilience and emotional strength, the writer could benefit from guided meditation and personal mentoring focused on emotional intelligence."

        Write the analysis in a professional, conversational tone, delivering actionable advice that the user can implement in their life. Focus on clarity, empathy, and relevance to their goals.
        """

    def save_result(self, result: Dict[str, Any]) -> None:
        """Save analysis result to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = Path(f"data/results/{timestamp}.json")

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            logger.info(f"Results saved to {file_path}")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise


# Initialize analyzer
analyzer = HandwritingAnalyzer()


@app.post("/analyze")
async def analyze_handwriting(file: UploadFile = File(...)):
    """Endpoint for handwriting analysis"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Analyze image
        result = await analyzer.analyze_image(file)
        return result

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main page"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend not found")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {str(exc)}")
    return {
        "status": "error",
        "message": str(exc),
        "timestamp": datetime.now().isoformat()
    }


def validate_image(image: np.ndarray) -> bool:
    """Validate image for analysis"""
    try:
        # Check image dimensions
        if image.size == 0 or len(image.shape) not in [2, 3]:
            return False

        # Check image content
        if np.all(image == 0) or np.all(image == 255):
            return False

        # Check minimum size
        if min(image.shape[:2]) < 100:
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        return False


class Config:
    """Application configuration"""
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    HOST = os.getenv("HOST", "localhost")
    PORT = int(os.getenv("PORT", 8001))
    MODEL_PATH = os.getenv("MODEL_PATH", "models")
    RESULTS_PATH = os.getenv("RESULTS_PATH", "data/results")
    UPLOAD_PATH = os.getenv("UPLOAD_PATH", "data/uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO if Config.DEBUG else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/app.log')
        ]
    )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Create necessary directories
    for path in [Config.MODEL_PATH, Config.RESULTS_PATH, Config.UPLOAD_PATH, 'logs']:
        Path(path).mkdir(parents=True, exist_ok=True)

    # Start server
    uvicorn.run(
        "app:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG,
        log_level="info" if Config.DEBUG else "warning"
    )
