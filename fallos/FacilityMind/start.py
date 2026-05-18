import os
import subprocess
import sys
import time

def main():
    print("\n" + "="*50)
    print("🚀 Starting FacilityMind Platform (Local Mode)...")
    print("="*50 + "\n")
    
    # Explicitly copy environment variables
    env = os.environ.copy()
    
    # 1. Start FastAPI Backend
    print("▶️ Starting Backend (FastAPI)...")
    backend = subprocess.Popen([sys.executable, "app/main.py"], env=env)
    
    # Give time for the local server to wake up
    time.sleep(3)
    
    # 2. Start Streamlit UI
    print("▶️ Starting Interface (Streamlit)...")
    frontend = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/ui.py"], env=env)
    
    print("\n" + "="*50)
    print(f"✅ All services online!")
    print(f"🔗 Streamlit Panel: http://127.0.0.1:8501")
    print("Press Ctrl+C to shut down all services together.")
    print("="*50 + "\n")
    
    try:
        # Keep main thread alive
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down all services...")
        backend.terminate()
        frontend.terminate()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()
