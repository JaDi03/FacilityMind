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
    
    # 1. Start Lobster Trap Security Proxy
    print("▶️ Starting Lobster Trap Security Proxy...")
    lobstertrap_path = os.path.join(os.path.dirname(__file__), "lobstertrap.exe")
    policy_path = os.path.join(os.path.dirname(__file__), "configs", "facilitymind_policy.yaml")
    security_proxy = None
    if os.path.exists(lobstertrap_path):
        security_proxy = subprocess.Popen([
            lobstertrap_path, "serve",
            "--policy", policy_path,
            "--listen", ":8080"
        ], env=env)
    else:
        print("⚠️ Warning: lobstertrap.exe not found. Running without security proxy dashboard.")
    
    # 2. Start FastAPI Backend
    print("▶️ Starting Backend (FastAPI)...")
    backend = subprocess.Popen([sys.executable, "app/main.py"], env=env)
    
    # Give time for the local servers to wake up
    time.sleep(3)
    
    # 3. Start Streamlit UI
    print("▶️ Starting Interface (Streamlit)...")
    frontend = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/ui.py"], env=env)
    
    print("\n" + "="*50)
    print(f"✅ All services online!")
    print(f"🔗 Streamlit Panel: http://127.0.0.1:8501")
    if security_proxy:
        print(f"🛡️ Lobster Trap Dashboard: http://127.0.0.1:8080/_lobstertrap/")
    print("Press Ctrl+C to shut down all services together.")
    print("="*50 + "\n")
    
    try:
        # Keep main thread alive
        backend.wait()
        frontend.wait()
        if security_proxy:
            security_proxy.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down all services...")
        backend.terminate()
        frontend.terminate()
        if security_proxy:
            security_proxy.terminate()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()
