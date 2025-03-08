# Safety Regulations Detector

This is an intelligent safety detection system based on computer vision and large language models that can automatically identify and analyze violations in images. The main functions include:

1. **Real-time Violation Detection**: Analyzes input images through computer vision models to identify potential violations
2. **Multi-scene Adaptation**: Supports safety detection across various scenarios including factories, construction sites, roads, etc.
3. **Intelligent Analysis Reports**: Generates detailed violation analysis reports by combining large language models

## Key Features

- High-precision object detection and behavior recognition
- Real-time alerts and notification system
- Customizable rules engine
- User-friendly interface
- Complete API support
- Detailed analysis report generation

## Application Scenarios

- Industrial safety monitoring
- Construction site safety management  
- Traffic safety monitoring
- Public space safety maintenance

The core logic, defined in `src/agent/graph.py`.

## Getting Started

Assuming you have already [installed Python Env], to set up:

1. Create a `.env` file.

```bash
cp .env.example .env
```

2. Define required API keys in your `.env` file.

3. Install dependencies.

```bash
pip install -e .
```

4. Run project.
```bash
python -m agent.ui
````