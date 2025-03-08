import base64
from PIL import Image, ImageDraw
from io import BytesIO
from agent.state import Coordinates
import requests


def mark_bbox_on_image(image_base64: str, coordinates_list: list[Coordinates], source: str = "web", mark_color=(0, 255, 0), mark_thickness=2):
    """Mark multiple bounding boxes on the image.
    
    Args:
        image_base64: Base64 encoded image string or image URL if source is "web"
        coordinates_list: List of Coordinates objects, each containing coordinates [x_min, y_min, x_max, y_max]
        source: Source type of image, either "web" or "local", defaults to "web"
        mark_color: RGB color tuple for the bounding box, defaults to green (0,255,0)
        mark_thickness: Thickness of the bounding box lines, defaults to 2
        
    Returns:
        Base64 encoded string of marked image, or empty string if error occurs
    """
    try:
        if source == "web":
            # For web source, convert URL to base64 first
            response = requests.get(image_base64)
            image_data = response.content
            image_base64 = base64.b64encode(image_data).decode()
        
        # Handle base64 string, remove data:image prefix if present
        if "base64," in image_base64:
            image_base64 = image_base64.split("base64,")[1]    

        # Decode base64 to image
        image = Image.open(BytesIO(base64.b64decode(image_base64)))
        draw = ImageDraw.Draw(image)
        
        # Draw each bounding box
        for coordinates in coordinates_list:
            x_min, y_min, x_max, y_max = coordinates.coordinates[0], coordinates.coordinates[1], coordinates.coordinates[2], coordinates.coordinates[3]
            
            # Draw rectangle with specified thickness
            for i in range(mark_thickness):
                draw.rectangle(
                    [x_min-i, y_min-i, x_max+i, y_max+i],
                    outline=mark_color
                )
        
        # Convert back to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        marked_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        return marked_image_base64
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return ""