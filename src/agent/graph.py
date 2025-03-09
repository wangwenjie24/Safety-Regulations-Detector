import os

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
    rule_writer_instructions_formatted = rule_writer_instructions.format(safety_items=state["safety_items"])
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
        Send("detect_image", {"rule": r.item, "image_url": state["image_url"]}) 
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

def compile_detection_result(state: DetectionState):
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
    return {"completed_detection_results": [detection_result]}

def should_locate(state: AnalysisState, config: RunnableConfig) -> Literal["get_targets", "compile_detection_result"]:
    """Route to next node based on check results."""
    configuration = Configuration.from_runnable_config(config)
    if configuration.need_mark and state["detection"].conclusion:
        return "get_targets"
    else:
        return "compile_detection_result"

# Add the node to the graph
detect_bulider = StateGraph(DetectionState, output=DetectionOutputState, config_schema=Configuration)
detect_bulider.add_node("detect", detect)
detect_bulider.add_node("get_targets", get_targets)
detect_bulider.add_node("locate", locate)
detect_bulider.add_node("compile_detection_result", compile_detection_result)

# Set the edges
detect_bulider.add_edge("__start__", "detect")
detect_bulider.add_conditional_edges("detect", should_locate)
detect_bulider.add_conditional_edges("get_targets", initiate_parallel_locate,  ["locate"])
detect_bulider.add_edge("locate", "compile_detection_result")
detect_bulider.add_edge("compile_detection_result", "__end__")

def gateher_data(state: AnalysisState):
    """Gather data from detection results."""
    # Get state
    completed_detection_results = state["completed_detection_results"]
    # Get violation detections
    coordinates_list = []
    explanation_list = []
    conclusion = False
    for detection_result in completed_detection_results:
        explanation_list.append(detection_result.explanation)
        coordinates_list.extend(detection_result.coordinates)
        if detection_result.conclusion:
            conclusion = True
    return {"coordinates_list": coordinates_list[1:], "explanation_list": explanation_list, "conclusion": conclusion}

def box(state: AnalysisState, config: RunnableConfig):
    """Box the detection."""
    configuration = Configuration.from_runnable_config(config)
    # Get state
    image_url = state["image_url"]
    coordinates_list = state["coordinates_list"]
    # Mark the bounding boxes
    marked_image_base64 = mark_bbox_on_image(image_url, coordinates_list, source=configuration.source, mark_thickness=4, mark_color=(255, 0, 0))
    return {"marked_image": marked_image_base64}

def should_box(state: AnalysisState, config: RunnableConfig) -> Literal["box", "finalize_result"]:
    """Determine if the detection result should be boxed."""
    configuration = Configuration.from_runnable_config(config)
    # Get state
    conclusion = state["conclusion"]
    # Determine if the detection result should be boxed
    if configuration.need_mark and conclusion:
        return "box"
    else:
        return "finalize_result"

def finalize_result(state: AnalysisState, config: RunnableConfig):
    """Finalize the result."""
    configuration = Configuration.from_runnable_config(config)
    # Get state
    conclusion = state["conclusion"]
    explanation_list = state["explanation_list"]
    coordinates_list = state["coordinates_list"]
    marked_image = state.get("marked_image", "") or ""
    # Format the result
    final_result = f"## Final Result\n\nConclusion: {conclusion} \n\n"
    final_result += "Explanation： \n"
    for explanation in explanation_list:
        final_result += f"- {explanation}\n"
    if conclusion and configuration.need_mark:
        final_result += "\nCoordinates:\n"
        for coordinates in coordinates_list:
            final_result += f"- {coordinates.coordinates}\n"
    return {"final_result": final_result, "marked_image": marked_image}

# Define the graph
workflow = StateGraph(AnalysisState, input=AnalysisInputState, output=AnalysisOutputState, config_schema=Configuration)

# Add the node to the graph
workflow.add_node("generate_rules", generate_rules)
workflow.add_node("detect_image", detect_bulider.compile())
workflow.add_node("gateher_data", gateher_data  )
workflow.add_node("box", box)
workflow.add_node("finalize_result", finalize_result)

# Set the edges
workflow.add_edge("__start__", "generate_rules")
workflow.add_conditional_edges("generate_rules", initiate_parallel_detection, ["detect_image"])
workflow.add_edge("detect_image", "gateher_data")
workflow.add_conditional_edges("gateher_data", should_box)
workflow.add_edge("box", "finalize_result")
workflow.add_edge("finalize_result", "__end__")

# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "safety-regulations-detector"