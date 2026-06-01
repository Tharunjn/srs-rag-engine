
import base64
import logging
import requests
from pathlib import Path

# Setup logging
log_file = Path("./extraction_debug.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)



def get_vlm_summary(image_bytes_list, context_text="", vlm_url="http://10.117.100.61:11434/api/chat", model="qwen3-vl:8b"):
    """
    Accepts a list of image bytes and a context string.
    For each image, sends a strong JSON prompt to the VLM and expects a JSON response with caption, description, and ui_elements.
    Returns a dict with keys: caption, description, ui_elements (for group use in image_processing.py).
    """
    results = {"caption": "", "description": "", "ui_elements": []}
    import json as _json
    if len(image_bytes_list) == 1:
        # Single image, same as before
        img_b64 = base64.b64encode(image_bytes_list[0]).decode('utf-8')
        prompt = (
            "You are an expert UI/UX analyst. Analyze the following image, which is part of a sequence of related images from a document. Here is the context from the document:\n"
            f"<context>\n{context_text.strip()}\n</context>\n\n"
            "For this image, reply STRICTLY in JSON with the following keys:\n"
            "- caption: A short, meaningful caption for the image (max 15 words).\n"
            "- description: A detailed, clear description of what this image shows and how it relates to the context (max 80 words).\n"
            "- ui_elements: A list of key UI elements or visual features present in the image (as a JSON array of strings).\n\n"
            "Example JSON:\n"
            "{\n  \"caption\": \"Login screen with username and password fields\",\n"
            "  \"description\": \"This image shows a login screen where users can enter their credentials to access the system. The context suggests it is the entry point for the application.\",\n"
            "  \"ui_elements\": [\"username field\", \"password field\", \"login button\"]\n}"
            "\n\nDO NOT include any commentary or explanation outside the JSON. Only output the JSON object."
        )
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64]
                }
            ],
            "stream": False
        }
        try:
            logger.info(f"Getting VLM summary for single image with context ({len(context_text)} chars)...")
            response = requests.post(vlm_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            content = result['message']['content']
            parsed = _json.loads(content)
            results["caption"] = parsed.get("caption", "")
            results["description"] = parsed.get("description", "")
            results["ui_elements"] = parsed.get("ui_elements", [])
        except Exception as e:
            logger.error(f"Error getting VLM summary for single image: {e}")
            results["description"] = "Image summary not available."
        return results
    # Multiple images: send all in one call, ask for group analysis
    images_b64 = [base64.b64encode(img).decode('utf-8') for img in image_bytes_list]
    prompt = (
        "You are an expert UI/UX analyst. Analyze the following sequence of images, which are consecutive and represent a connected UI or flow. Here is the context from the document:\n"
        f"<context>\n{context_text.strip()}\n</context>\n\n"
        "For this group of images, reply STRICTLY in JSON with the following keys:\n"
        "- caption: A short, meaningful caption for the group (max 15 words).\n"
        "- description: A detailed, clear description of what the group of images shows and how it relates to the context (max 80 words).\n"
        "- ui_elements: A list of key UI elements or visual features present in the group (as a JSON array of strings).\n\n"
        "Example JSON:\n"
        "{\n  \"caption\": \"Multi-step login and dashboard flow\",\n"
        "  \"description\": \"These images show a login process followed by a dashboard. The context suggests a user journey from authentication to main app features.\",\n"
        "  \"ui_elements\": [\"login form\", \"dashboard widgets\", \"navigation bar\"]\n}"
        "\n\nDO NOT include any commentary or explanation outside the JSON. Only output the JSON object."
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64
            }
        ],
        "stream": False
    }
    try:
        logger.info(f"Getting VLM summary for image group of {len(image_bytes_list)} images with context ({len(context_text)} chars)...")
        response = requests.post(vlm_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        content = result['message']['content']
        parsed = _json.loads(content)
        results["caption"] = parsed.get("caption", "")
        results["description"] = parsed.get("description", "")
        results["ui_elements"] = parsed.get("ui_elements", [])
    except Exception as e:
        logger.error(f"Error getting VLM summary for image group: {e}")
        results["description"] = "Image group summary not available."
    return results

    