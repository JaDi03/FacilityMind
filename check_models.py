import os
import google.generativeai as genai
from dotenv import load_dotenv

def list_design_models():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
        return

    genai.configure(api_key=api_key)

    print("\n--- Checking available models for your API Key ---\n")
    
    # 1. List ALL models
    try:
        all_models = list(genai.list_models())
        print(f"Total models accessible: {len(all_models)}")
    except Exception as e:
        print(f"Error listing models: {e}")
        return
    
    # 2. Filter for Image Generation models
    print("\nMODELS FOR DESIGN (Image Generation):")
    design_models = [m for m in all_models if 'generateImage' in str(m.supported_generation_methods) or 'image' in m.name.lower()]
    
    if design_models:
        for m in design_models:
            print(f"  - {m.name}")
    else:
        print("  - No native 'generateImage' models found in this list.")

    # 3. Filter for High-Reasoning models
    print("\nMODELS FOR REASONING (Pro/Ultra):")
    reasoning_models = [m for m in all_models if 'pro' in m.name.lower() or 'ultra' in m.name.lower()]
    for m in reasoning_models:
        print(f"  - {m.name}")

    # 4. Filter for Embeddings
    print("\nMODELS FOR EMBEDDINGS:")
    embed_models = [m for m in all_models if 'embedContent' in m.supported_generation_methods]
    for m in embed_models:
        print(f"  - {m.name}")

    print("\n--- Diagnostic Complete ---")

if __name__ == "__main__":
    list_design_models()
