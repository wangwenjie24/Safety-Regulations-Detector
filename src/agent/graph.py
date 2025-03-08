import os
import base64
from dotenv import load_dotenv
from typing_extensions import Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.constants import Send

from agent.configuration import Configuration
from agent.state import (
    AnalysisState, 
    AnalysisInputState,
    AnalysisOutputState,
    Detection,
    DetectionOutputState,
    Rules,
    DetectionState,
    DetctionResult,
    Coordinates
)
from agent.prompts import rule_writer_instructions, detector_instructions, locator_instructions
from agent.utils import mark_bbox_on_image

load_dotenv()

def generate_rules(state: AnalysisState):
    """Generate safety regulation rules."""
    # Format the prompt
    rule_writer_instructions_formatted = rule_writer_instructions.format(description=state["description"])
    # Call LLM
    llm_with_structured_output = ChatOpenAI(
        model_name=os.getenv("RULE_WRITER_MODEL"),
        openai_api_base=os.getenv("OPENAI_BASE_URL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"), 
        temperature=0.0
    ).with_structured_output(Rules)
    # Invoke the LLM
    result = llm_with_structured_output.invoke([
        SystemMessage(content=rule_writer_instructions_formatted),
        HumanMessage(content="Please generate safety regulation rules.")
    ])
    return {"rules": result.rules}

def initiate_parallel_detection(state: AnalysisState):
    """Detect each query using the Send API to parallelize the process."""
    return [
        Send("image_detection", {"rule": r.item, "image_url": state["image_url"]}) 
        for r in state["rules"]
        if r.item
    ]

def detect(state: DetectionState):
    """Detect safety violations in the image based on the rule."""
    # Get state
    image_url = state["image_url"]
    rule = state["rule"]
    # Format the prompt
    detector_instructions_formatted = detector_instructions.format(rule=rule)
    # Call LLM
    try:
        llm_with_structured_output = ChatOpenAI(
            model_name=os.getenv("DETECTOR_MODEL"),
            openai_api_base=os.getenv("OPENAI_BASE_URL"), 
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0
        ).with_structured_output(Detection)
        # Invoke the LLM
        detection = llm_with_structured_output.invoke([
            SystemMessage(content=detector_instructions_formatted),
            HumanMessage(content=[
                {
                    "type": "image_url", 
                    "min_pixels": 128*28*28,
                    "max_pixels": 4096*28*28,
                    "image_url": {
                        "url": image_url
                    }
                },
                {"type": "text", "text": "Please analyze this image."}
            ])
        ])
        return {"detection": detection, "coordinates": [Coordinates(coordinates=[0, 0, 0, 0])]}
    except Exception as e:
        print(f"Error in detection: {e}")
        # Return empty detection result on error
        return {
            "detection": Detection(
                conclusion=False,
                explanation="Error occurred during detection",
                targets=[]
            ),
            "coordinates": [Coordinates(coordinates=[0, 0, 0, 0])]
        }

def get_targets(state: DetectionState):
    """Get the target by the description."""
    return {"targets": state["detection"].targets}

def initiate_parallel_locate(state: DetectionState):
    """Locate each target using the Send API to parallelize the process."""
    return [
        Send("locate", {"target": t, "image_url": state["image_url"]}) 
        for t in state["targets"]
        if t
    ]

def locate(state: DetectionState):
    """Get the coordinates by the description."""
    # Get state
    image_url = state["image_url"]
    target = state["target"]
    # Format the prompt
    locator_instructions_formatted = locator_instructions.format(description=target)
    # Call LLM
    llm_with_structured_output = ChatOpenAI(
        model_name=os.getenv("LOCATOR_MODEL"), 
        openai_api_base=os.getenv("OPENAI_BASE_URL"), 
        openai_api_key=os.getenv("OPENAI_API_KEY"), 
        temperature=0.0
    ).with_structured_output(Coordinates)
    # Invoke the LLM
    coordinates = llm_with_structured_output.invoke([
        SystemMessage(content=locator_instructions_formatted),
        HumanMessage(content=[
            {
                "type": "image_url",
                "min_pixels": 128*28*28,
                "max_pixels": 4096*28*28,
                "image_url": {
                    "url": image_url
                }
            },
            {"type": "text", "text": "Get the locations in the form of coordinates."}
        ])
    ])
    return {"coordinates": [coordinates]}

def compile_data(state: DetectionState):
    """Compile the detection result."""
    # Get state
    detection = state["detection"]
    coordinates = state["coordinates"]
    detection_result = DetctionResult(
        conclusion=detection.conclusion,
        targets=detection.targets,
        explanation=detection.explanation,
        coordinates=coordinates
    )
    return {"completed_detections": [detection_result]}

def should_locate(state: AnalysisState, config: RunnableConfig) -> Literal["get_targets", "compile_data"]:
    """Route to next node based on check results."""
    configuration = Configuration.from_runnable_config(config)
    if configuration.need_mark and state["detection"].conclusion:
        return "get_targets"
    else:
        return "compile_data"

def gateher_violations(state: AnalysisState):
    """Gather violation detections from detection."""
    # Get state
    completed_detections = state["completed_detections"]
    # Get violation detections
    violation_detections = []
    for detection in completed_detections:
        if detection.conclusion:
            violation_detections.append(detection)
    return {"violation_detections": violation_detections}

def plots_bounding_boxes(state: AnalysisState, config: RunnableConfig):
    """Box the detection."""
    configuration = Configuration.from_runnable_config(config)
    # Get state
    image_url = state["image_url"]
    violation_detections = state["violation_detections"]
    # Mark the bounding boxes
    coordinates_list = []
    for detection in violation_detections:
        coordinates_list.extend(detection.coordinates)
    marked_image_base64 = mark_bbox_on_image(image_url, coordinates_list[1:], source=configuration.source, mark_thickness=4, mark_color=(255, 0, 0))
    return {"marked_image": marked_image_base64}

def should_plots(state: AnalysisState, config: RunnableConfig) -> Literal["plots_bounding_boxes", "finalize_result"]:
    """Determine if the detection result should be boxed."""
    configuration = Configuration.from_runnable_config(config)
    if configuration.need_mark and len(state["violation_detections"]) > 0:
        return "plots_bounding_boxes"
    else:
        return "finalize_result"

def finalize_result(state: AnalysisState):
    """Finalize the result."""
    conclusion = True if len(state["violation_detections"]) > 0 else False
    final_result = f"## Final Result\n\nConclusion: {conclusion} \n\n"
    
    if len(state["violation_detections"]) > 0:
        final_result += "Explanation： \n"
        for detection in state["violation_detections"]:
            final_result += f"- {detection.explanation}\n"
        
        final_result += "\nCoordinates:\n"
        for detection in state["violation_detections"]:
            for coordinates in detection.coordinates[1:]:
                final_result += f"- {coordinates.coordinates}\n"

 
    marked_image = state.get("marked_image", "") or ""

    return {"final_result": final_result, "marked_image": marked_image}

# Add the node to the graph
detect_bulider = StateGraph(DetectionState, output=DetectionOutputState, config_schema=Configuration)
detect_bulider.add_node("detect", detect)
detect_bulider.add_node("get_targets", get_targets)
detect_bulider.add_node("locate", locate)
detect_bulider.add_node("compile_data", compile_data)

# Set the edges
detect_bulider.add_edge("__start__", "detect")
detect_bulider.add_conditional_edges("detect", should_locate)
detect_bulider.add_conditional_edges("get_targets", initiate_parallel_locate,  ["locate"])
detect_bulider.add_edge("locate", "compile_data")
detect_bulider.add_edge("compile_data", "__end__")

# Define a new graph
workflow = StateGraph(AnalysisState, input=AnalysisInputState, output=AnalysisOutputState, config_schema=Configuration)

# Add the node to the graph
workflow.add_node("generate_rules", generate_rules)
workflow.add_node("image_detection", detect_bulider.compile())
workflow.add_node("gateher_violations", gateher_violations  )
workflow.add_node("plots_bounding_boxes", plots_bounding_boxes)
workflow.add_node("finalize_result", finalize_result)

# Set the edges
workflow.add_edge("__start__", "generate_rules")
workflow.add_conditional_edges("generate_rules", initiate_parallel_detection, ["image_detection"])
workflow.add_edge("image_detection", "gateher_violations")
workflow.add_conditional_edges("gateher_violations", should_plots)
workflow.add_edge("plots_bounding_boxes", "finalize_result")
workflow.add_edge("finalize_result", "__end__")

# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "safety-regulations-detector"


# if __name__ == "__main__":

#     # Load the image from filepath
#     image_path = "src/agent/47268ab687fe862aa2ca4f795a772d3a.jpg"
#     with open(image_path, "rb") as image_file:
#         image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

#     image = f"data:image/jpeg;base64,{image_base64}"
#     result = graph.invoke({"description":"检查是否佩戴安全帽", "image_url": image}, {"configurable": {"need_mark": True, "source": "local"}})
#     print(result)