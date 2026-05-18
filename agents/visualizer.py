"""
FacilityMind — Agent 4: Visualizer
Generates 2D technical visualizations of infrastructure paths using Gemini + Imagen 4.0 Ultra.
This agent provides visual aids to enhance technical understanding of the system's responses.
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional


from config import GEMINI_API_KEY, GeminiModels, VISUALIZATION_ENABLED
from models.schemas import ReasonerOutput, ValidatorOutput, VisualizationOutput

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "visualizer.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else ""

MODEL_TEXT = GeminiModels.REASONER  # Using Pro to generate technical image descriptions
MODEL_IMAGE = GeminiModels.VISUALIZER_IMAGE  # Imagen 4.0 Ultra for image generation


async def agente_visualizador(
    reasoner: ReasonerOutput,
    validator: ValidatorOutput,
    perception_data: dict,
    output_dir: str = "./data/processed/visualizations"
) -> VisualizationOutput:
    """
    Visualizer Agent: Generates technical images and diagrams of the traced circuit or path.

    Args:
        reasoner: Output from the Reasoner Agent (contains traced circuit data).
        validator: Output from the Validator Agent.
        perception_data: Perception data for visual context.
        output_dir: Directory where generated visualizations are saved.

    Returns:
        VisualizationOutput containing descriptions, diagrams, and image paths.
    """
    logger.info(f"[Visualizer] Starting visualization generation | enabled={VISUALIZATION_ENABLED}")

    if not VISUALIZATION_ENABLED:
        return VisualizationOutput(
            visual_summary="Visualization is disabled in the system configuration."
        )

    # Skip visualization if no circuit path was traced
    if not reasoner.traced_circuit or not reasoner.traced_circuit.nodes:
        logger.info("[Visualizer] No circuit path traced — skipping visualization generation")
        return VisualizationOutput(
            visual_summary=reasoner.candidate_response[:200] + "..." if len(reasoner.candidate_response) > 200 else reasoner.candidate_response
        )

    # --- 1. Generate Technical Image Description ---
    descripcion_imagen = await _generar_descripcion_imagen(reasoner, validator, perception_data)

    # --- 2. Generate Mermaid Diagram ---
    diagrama_mermaid = _generar_diagrama_mermaid(reasoner.traced_circuit)

    # --- 3. Generate Image via Imagen 4.0 Ultra ---
    # User requested to disable image generation from scratch as it hallucinates CAD drawings.
    imagen_path = None

    # --- 4. Generate Visual Summary ---
    resumen = _generar_resumen_visual(reasoner)

    output = VisualizationOutput(
        generated_image=imagen_path,
        image_description=descripcion_imagen,
        mermaid_diagram=diagrama_mermaid,
        visual_summary=resumen,
        heatmap=None  # Future work: building heatmap generation
    )

    logger.info(f"[Visualizer] Visualization complete | image_generated={imagen_path is not None}, mermaid_ready={diagrama_mermaid is not None}")
    return output


async def _generar_descripcion_imagen(
    reasoner: ReasonerOutput,
    validator: ValidatorOutput,
    perception_data: dict
) -> str:
    """Generates detailed prompt for Imagen 4.0 Ultra using Gemini Pro."""

    circuito = reasoner.traced_circuit
    circuito_json = json.dumps(circuito.model_dump(), ensure_ascii=False, indent=2) if circuito else "null"

    prompt = f"""You are a technical visualization expert.
TRACED CIRCUIT: {circuito_json}
VALIDATED RESPONSE: {validator.final_response[:500]}

Generate a detailed English description for Imagen 4.0. The goal is to generate a PHOTOREALISTIC or 3D ISOMETRIC representation of the physical space described in the response, NOT a 2D floorplan.
The user already has the 2D floorplan. We want to show them what this looks like in real life.
The description must:
1. Specify style: Photorealistic, highly detailed 3D facility environment, professional lighting.
2. Clearly describe the physical components installed (e.g., pendant lights over an island, electrical panels).
3. Do NOT mention words like 'CAD', 'blueprint', 'diagram', or '2D'.
4. Be maximum 150 words.

OUTPUT: Return ONLY a JSON object with:
{{"descripcion": "image prompt text", "resumen": "short summary"}}
"""

    try:
        from google import genai as genai_v2
        client = genai_v2.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1beta'})
        response = await client.aio.models.generate_content(
            model=MODEL_TEXT,
            contents=prompt
        )
        raw = response.text.replace("```json", "").replace("```", "").strip()

        data = json.loads(raw)
        return data.get("descripcion", _fallback_descripcion(circuito))
    except Exception as e:
        logger.warning(f"[Visualizer] Error generating technical description: {e}")
        return _fallback_descripcion(circuito)


def _fallback_descripcion(circuito) -> str:
    """Fallback description if Gemini fails to generate a dynamic one."""
    if not circuito or not circuito.nodes:
        return "Technical diagram of building infrastructure system."

    nodos_str = " → ".join(circuito.nodes)
    tipo = circuito.type or "infrastructure"

    return (
        f"A photorealistic, highly detailed 3D rendering of a building {tipo} system installation. "
        f"It shows the physical space with the following components installed in real life: {nodos_str}. "
        f"Do NOT generate a 2D floorplan or CAD drawing. Generate a beautiful, realistic visualization of what this infrastructure looks like installed in the facility."
    )


def _generar_diagrama_mermaid(circuito) -> Optional[str]:
    """Generates a Mermaid.js diagram for the traced circuit path."""
    if not circuito or not circuito.nodes:
        return None

    try:
        nodos = circuito.nodes
        tipo = circuito.type or "general"

        # Determine diagram orientation
        direction = "LR" if len(nodos) <= 5 else "TD"

        lines = [f"graph {direction}"]

        for i, nodo in enumerate(nodos):
            # Sanitize label for Mermaid syntax to prevent crashes
            label = nodo.replace('"', "'").replace("\n", " ").strip()
            # Truncate if excessively long
            if len(label) > 60:
                label = label[:57] + "..."
            
            # Wrap in double quotes to allow special characters like () or :
            lines.append(f'    N{i}["{label}"]')

        # Define node connections
        for i in range(len(nodos) - 1):
            lines.append(f"    N{i} --> N{i+1}")

        # Apply styling to start and end nodes
        lines.append(f"    style N0 fill:#e1f5fe,stroke:#333,stroke-width:2px")
        lines.append(f"    style N{len(nodos)-1} fill:#fff9c4,stroke:#333,stroke-width:2px")

        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"[Visualizer] Failed to generate Mermaid diagram: {e}")
        return None


async def _generar_imagen_imagen4(descripcion: str, output_dir: str) -> Optional[str]:
    """
    Generates an image using Imagen 4.0 Ultra via the new google-genai SDK.
    """
    try:
        from google import genai as genai_v2
        from google.genai import types

        os.makedirs(output_dir, exist_ok=True)

        client = genai_v2.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1beta'})

        # Generate unique filename
        import hashlib
        import time
        filename_hash = hashlib.md5(f"{descripcion}{time.time()}".encode()).hexdigest()[:8]
        output_path = os.path.join(output_dir, f"path_{filename_hash}.png")

        # Call Imagen 4.0 Ultra API
        response = await asyncio.to_thread(
            client.models.generate_images,
            model=MODEL_IMAGE,
            prompt=descripcion[:300],
            config=types.GenerateImagesConfig(
                number_of_images=1,
                include_rai_reason=True,
                output_mime_type="image/png"
            )
        )

        if response and response.generated_images:
            image_data = response.generated_images[0].image.image_bytes
            with open(output_path, "wb") as f:
                f.write(image_data)
            logger.info(f"[Visualizer] Image saved successfully via google-genai: {output_path}")
            return output_path

        return None

    except Exception as e:
        logger.warning(f"[Visualizer] Error communicating with Imagen 4.0 via SDK: {e}")
        return await _generar_imagen_rest_api(descripcion, output_dir)


async def _generar_imagen_rest_api(descripcion: str, output_dir: str) -> Optional[str]:
    """Fallback: Invokes the Imagen API directly via REST."""
    import aiohttp

    try:
        os.makedirs(output_dir, exist_ok=True)

        import hashlib
        import time
        filename_hash = hashlib.md5(f"{descripcion}{time.time()}".encode()).hexdigest()[:8]
        output_path = os.path.join(output_dir, f"path_{filename_hash}.png")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_IMAGE}:generateImage"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        }
        payload = {
            "instances": [{"prompt": descripcion[:300]}],
            "parameters": {"sampleCount": 1}
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    if result.get("predictions"):
                        import base64
                        image_b64 = result["predictions"][0].get("bytesBase64Encoded", "")
                        if image_b64:
                            with open(output_path, "wb") as f:
                                f.write(base64.b64decode(image_b64))
                            logger.info(f"[Visualizer] Image generated via REST API: {output_path}")
                            return output_path

        return None

    except Exception as e:
        logger.warning(f"[Visualizer] REST API fallback failed: {e}")
        return None


def _generar_resumen_visual(reasoner: ReasonerOutput) -> str:
    """Generates a plain-text visual summary of the traced path."""
    if not reasoner.traced_circuit:
        return reasoner.candidate_response[:250]

    nodos = reasoner.traced_circuit.nodes
    if not nodos:
        return reasoner.candidate_response[:250]

    # Show up to 6 nodes in the summary
    resumen = " → ".join(nodos[:6])
    if len(nodos) > 6:
        resumen += f" → ... ({len(nodos) - 6} more)"

    return resumen
