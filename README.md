# FacilityMind: Professional Multi-Agent System for Facility Management

FacilityMind is an advanced, enterprise-grade multi-agent AI system designed to streamline facility management operations by providing technicians with instantaneous, verified access to technical documentation and building blueprints.

By leveraging state-of-the-art Large Language Models (LLMs) and a robust multi-agent architecture, FacilityMind transforms thousands of pages of complex blueprints into an interactive, conversational technical resource.

## Core Features

- **Multimodal Interaction**: Technicians can interact using voice, images, or text, enabling hands-free operation in the field.
- **Intelligent Blueprint Ingestion**: Automated indexing of PDF blueprints (Architectural, Electrical, Plumbing, HVAC, Structural) into a high-performance vector database.
- **Deep Technical Reasoning**: Traces complex circuits and infrastructure paths across multiple documentation pages using advanced RAG and Long-Context capabilities.
- **Automated Validation & Safety**: A dedicated validator agent cross-checks all technical responses against original documentation to eliminate hallucinations and assess safety risks (e.g., LOTO requirements, emergency system impacts).
- **Automated Visualizations**: Generates 2D technical diagrams and Mermaid.js flowcharts to illustrate technical solutions and circuit paths.

## System Architecture

FacilityMind utilizes a sophisticated four-agent pipeline to ensure maximum accuracy and safety:

1.  **Perception Agent**: Processes multimodal field inputs (audio/image/text) to extract structured technical requirements, location data, and urgency levels.
2.  **Technical Reasoner**: Conducts deep-context retrieval and reasoning across the blueprint database to generate detailed technical solutions with exact citations.
3.  **Validator Agent**: Performs rigorous cross-verification of the reasoner's output against the original source text. It detects technical contradictions and evaluates operational safety risks.
4.  **Visualizer Agent**: Translates the validated technical response into clear visual aids, including circuit diagrams and process flowcharts.

## Technical Stack

- **Core Intelligence**: Google Gemini 2.0 (Pro/Flash) & Imagen 4.0 Ultra
- **Knowledge Base**: ChromaDB (Vector Store) with Gemini-native Embeddings (`text-embedding-004`)
- **Orchestration**: FastAPI (Python)
- **Interface**: Streamlit-based Technical Dashboard
- **Document Processing**: Advanced PDF extraction using PyMuPDF

## Installation & Deployment

### Prerequisites
- Python 3.10 or higher
- Google Gemini API Key

### Setup
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/organization/facilitymind.git
    cd facilitymind
    ```

2.  **Initialize environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```

3.  **Configuration**:
    Create a `.env` file based on `.env.example` and provide your `GEMINI_API_KEY`.

4.  **Launch Services**:
    - **Backend (API)**: `python -m app.main`
    - **Frontend (UI)**: `streamlit run app/ui.py`

## Data Ingestion

To index new blueprints into the system, use the provided ingestion utility:

```bash
python ingest_planos.py --dir path/to/blueprints/ --edificio BUILDING_ID
```

## Security and Safety

FacilityMind is designed with a safety-first approach. The system automatically identifies high-risk operations and flags them for supervisor approval. It enforces Lock-Out/Tag-Out (LOTO) awareness and monitors for impacts on critical infrastructure such as fire suppression and emergency power systems.

## License

This project is licensed under the MIT License.
