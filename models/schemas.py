"""
FacilityMind — Pydantic Data Schemas
Defines the data structures flowing between agents in the multi-agent system.
Each agent produces typed output that serves as input for the next agent in the pipeline.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Citation of a specific source within building blueprints."""
    blueprint_id: str = Field(..., description="Blueprint ID (e.g., E-14, P-01, A-03)")
    page_number: int = Field(..., description="Page number within the PDF")
    cited_text: str = Field(..., description="Exact text or paragraph cited from the blueprint")
    citation_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence that this citation is relevant")


class PerceptionOutput(BaseModel):
    """
    Output from Agent 1 — Perception.
    Analyzes audio + image + text from the field technician to extract structured data.
    """
    floor: Optional[str] = Field(None, description="Detected floor number (e.g., '14', 'GF', 'B1')")
    tower: Optional[str] = Field(None, description="Tower or wing identifier (e.g., '2', 'A', 'North')")
    room: Optional[str] = Field(None, description="Room or unit identifier (e.g., '14-B', '302')")
    detected_object: Optional[str] = Field(None, description="Identified object: electrical outlet, pipe, breaker, valve, etc.")
    query_objective: str = Field(..., description="Technician's objective (inferred from the query)")
    discipline: Optional[str] = Field(None, description="Inferred discipline: electrical, plumbing, HVAC, structural, general")
    urgency_notes: Optional[str] = Field(None, description="Urgency or risk keywords detected")
    detected_language: str = Field("en", description="Primary input language (en, es, etc.)")
    perception_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Overall perception confidence level (0-1)")
    intent_classification: str = Field("technical_query", description="Intent: technical_query, conversational, meta_query, off_topic, ambiguous, credential_extraction")
    raw_transcription: Optional[str] = Field(None, description="Raw audio transcription (if applicable)")


class TracedCircuit(BaseModel):
    """Represents a circuit or line traced through building blueprints."""
    nodes: List[str] = Field(default_factory=list, description="Ordered list of components (e.g., ['Outlet R-14-B-03', 'J-Box J-14-B-01', 'Circuit C-14-07', 'Panel PP-14-EAST', 'Breaker 23'])")
    description: str = Field("", description="Textual description of the traced path")
    type: str = Field("", description="Trace type: electrical, plumbing, HVAC, etc.")


class ReasonerOutput(BaseModel):
    """
    Output from Agent 2 — Technical Reasoner.
    Generates a candidate technical response based on RAG + Long Context.
    """
    candidate_response: str = Field(..., description="Complete and detailed technical response")
    sources: List[SourceCitation] = Field(default_factory=list, description="Cited blueprint sources")
    traced_circuit: Optional[TracedCircuit] = Field(None, description="Traced component path")
    recommended_steps: List[str] = Field(default_factory=list, description="Recommended action steps for the technician")
    mentioned_materials: List[str] = Field(default_factory=list, description="Materials or specifications mentioned in the response")
    cited_regulation: Optional[str] = Field(None, description="Regulatory codes or standards cited (NEC, ISO, etc.)")
    initial_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence of the candidate response")
    technical_warnings: List[str] = Field(default_factory=list, description="Technical warnings detected")
    rag_context: str = Field(default="", description="The RAG text used, passed forward for Validator fallback.")


class SafetyAssessment(BaseModel):
    """Safety evaluation of the candidate response."""
    electrical_risk: bool = Field(False, description="Involves electrical risk")
    plumbing_risk: bool = Field(False, description="Involves flood or gas risk")
    affects_emergency: bool = Field(False, description="Affects emergency systems")
    affects_elevators: bool = Field(False, description="Affects elevator systems")
    affects_alarms: bool = Field(False, description="Affects alarm or fire protection systems")
    requires_loto: bool = Field(False, description="Requires Lock-Out/Tag-Out (LOTO) procedure")
    risk_level: str = Field("low", description="low, medium, high, critical")
    risk_description: Optional[str] = Field(None, description="Explanation of identified risk")


class ValidatorOutput(BaseModel):
    """
    Output from Agent 3 — Validator.
    Verifies the candidate response against original blueprints.
    """
    approved: bool = Field(..., description="Whether the response passed validation")
    final_response: str = Field(..., description="Corrected or approved final response")
    final_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence after validation")
    verified_sources: List[SourceCitation] = Field(default_factory=list, description="Sources verified as accurate")
    warnings: List[str] = Field(default_factory=list, description="Warnings generated by the validator")
    requires_supervisor: bool = Field(False, description="Whether supervisor approval is required before proceeding")
    rejection_reason: Optional[str] = Field(None, description="Reason for rejection, if applicable")
    detected_hallucinations: List[str] = Field(default_factory=list, description="Detected fabricated data")
    safety: SafetyAssessment = Field(default_factory=SafetyAssessment, description="Safety assessment results")


class VisualizationOutput(BaseModel):
    """
    Output from Agent 4 — Visualizer.
    Generates images and diagrams to illustrate the response.
    """
    generated_image: Optional[str] = Field(None, description="Path or URL of the generated image (traced circuit)")
    image_description: Optional[str] = Field(None, description="Description of the visualization content")
    mermaid_diagram: Optional[str] = Field(None, description="Mermaid diagram for UI rendering")
    visual_summary: str = Field("", description="Visual summary of the response for the technician")
    heatmap: Optional[str] = Field(None, description="Path to building heatmap (if applicable)")


class FacilityMindResponse(BaseModel):
    """Complete final response from the orchestrated system."""
    response: str = Field(..., description="Response text for the technician")
    sources: List[SourceCitation] = Field(default_factory=list, description="Verified sources")
    traced_circuit: Optional[TracedCircuit] = Field(None, description="Traced circuit path, if applicable")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Final system confidence")
    warnings: List[str] = Field(default_factory=list, description="All accumulated warnings")
    requires_supervisor: bool = Field(False, description="Whether supervisor review is required")
    visualization: Optional[VisualizationOutput] = Field(None, description="Generated visualizations")
    safety: SafetyAssessment = Field(default_factory=SafetyAssessment, description="Safety assessment results")
    debug: dict = Field(default_factory=dict, description="Debug information from each agent")
    processing_time_ms: int = Field(0, description="Total processing time in milliseconds")


class PlanoMetadata(BaseModel):
    """Metadata for an ingested blueprint."""
    plano_id: str = Field(..., description="Unique blueprint ID (e.g., E-14, P-01)")
    tipo_plano: str = Field(..., description="electrical, plumbing, architectural, structural, general")
    edificio: str = Field("", description="Building Identifier")
    piso: Optional[str] = Field(None, description="Floor covered by the blueprint")
    torre: Optional[str] = Field(None, description="Tower or wing covered by the blueprint")
    total_paginas: int = Field(0, description="Total number of pages")
    fecha_ingesta: Optional[str] = None
    descripcion: Optional[str] = Field(None, description="Blueprint description")


class ChatMessage(BaseModel):
    """Chat message for conversational interaction."""
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = None
    tiene_imagen: bool = Field(False)
    tiene_audio: bool = Field(False)
