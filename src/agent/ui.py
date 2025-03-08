from typing import Dict, Any, Tuple
import gradio as gr
from PIL import Image
from io import BytesIO
import base64

from agent.graph import graph

def process_image(
    safety_items: list,  # 改为接收列表
    image_file: str,
    need_mark: bool,
    source: str
) -> Tuple[str, Image.Image]:
    """Process an image with safety inspection graph and return detection results.
    
    Args:
        safety_items: List of safety items to check
        image_file: Path to the image file to analyze
        need_mark: Whether to mark the detected issues on image
        source: Source type ('local' or 'web')
        
    Returns:
        Tuple containing:
        - Detection results as markdown formatted string
        - Processed image with markings (or original if no markings)
    """
    
    # 将选中的安全项目转换为描述文本
    description = "检查" + "，".join(safety_items)
    
    # Load and prepare input image
    with open(image_file, "rb") as f:
        image = Image.open(BytesIO(f.read()))
    
    # Convert image to base64 for API
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_url = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"

    # Run safety inspection through graph
    result = graph.invoke(
        {
            "description": description,
            "image_url": image_url
        },
        {"configurable": {"need_mark": need_mark, "source": source}}
    )
    
    # Process marked image if available
    marked_image = image  # Default to original image
    if result.get('marked_image'):
        try:
            img_data = result['marked_image']
            # Handle data URI format if present
            if "base64," in img_data:
                img_data = img_data.split("base64,")[1]
            
            marked_image = Image.open(BytesIO(base64.b64decode(img_data)))
        except Exception as e:
            print(f"Warning: Failed to process marked image: {e}")
            # Keep original image on error

    return result['final_result'], marked_image

# Create Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Safety Regulations Detector")
    
    with gr.Row():
        # Input column
        with gr.Column(scale=1):
            safety_items = gr.CheckboxGroup(
                choices=["佩戴安全帽", "玩手机", "吸烟"],
                label="Safety Items",
                info="Select items to check",
                value=["佩戴安全帽"]  # 默认选中安全帽
            )
            image_input = gr.File(
                label="Upload Image",
                file_types=["image"],
                type="filepath"
            )
            with gr.Row():
                need_mark = gr.Checkbox(
                    label="Mark Issues on Image",
                    value=True,
                    info="Whether to highlight detected issues on the image"
                )
                source = gr.Radio(
                    choices=["local", "web"],
                    value="local",
                    label="Source Type",
                    info="Select the source type for processing"
                )
            detect_btn = gr.Button("Detect")
        
        # Output column    
        with gr.Column(scale=1):
            text_output = gr.Markdown(label="Detection Results")
            image_output = gr.Image(label="Marked Image")
    
    # Wire up the interface
    detect_btn.click(
        fn=process_image,
        inputs=[safety_items, image_input, need_mark, source],
        outputs=[text_output, image_output]
    )

if __name__ == "__main__":
    demo.launch() 