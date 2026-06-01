"""
Quick Start RAG UI - Interactive Launcher
"""

import os
import sys
import subprocess

def print_banner():
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                      🔍 SRS RAG UI - QUICK LAUNCHER                           ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """)

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📋 Checking dependencies...")
    
    required = {
        "streamlit": "streamlit",
        "gradio": "gradio",
        "requests": "requests",
        "qdrant_client": "qdrant-client"
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True

def show_menu():
    """Display main menu"""
    print("\n" + "="*80)
    print("CHOOSE UI:")
    print("="*80)
    print("1️⃣  Streamlit (Recommended) - Modern, interactive, professional")
    print("2️⃣  Gradio - Lightweight, fast, shareable")
    print("3️⃣  Show System Info")
    print("4️⃣  Exit")
    print("-"*80)

def show_system_info():
    """Display system information"""
    print("\n" + "="*80)
    print("📊 SYSTEM INFORMATION")
    print("="*80)
    
    import requests
    
    OLLAMA_URL = "http://10.117.100.61:11434"
    QDRANT_URL = "http://10.188.105.70:6333"
    
    # Check Ollama
    print("\n🤖 Ollama Status:")
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"  ✅ Connected ({len(models)} models available)")
            for model in models[:5]:
                name = model.get("name", "").replace(":latest", "")
                print(f"     • {name}")
            if len(models) > 5:
                print(f"     ... and {len(models) - 5} more")
        else:
            print(f"  ⚠️  Connection failed (Status: {response.status_code})")
    except Exception as e:
        print(f"  ❌ Connection error: {e}")
    
    # Check Qdrant
    print("\n💾 Qdrant Status:")
    try:
        response = requests.get(f"{QDRANT_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"  ✅ Connected")
        else:
            print(f"  ⚠️  Connection failed (Status: {response.status_code})")
    except Exception as e:
        print(f"  ❌ Connection error: {e}")
    
    print("\n📍 URLs:")
    print(f"  Ollama: {OLLAMA_URL}")
    print(f"  Qdrant: {QDRANT_URL}")

def launch_streamlit():
    """Launch Streamlit app"""
    print("\n🚀 Launching Streamlit...")
    print("This will open your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app_streamlit.py"],
            cwd=os.path.dirname(__file__)
        )
    except KeyboardInterrupt:
        print("\n\n✅ Streamlit server stopped")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {e}")
        print("\nTry installing:")
        print("  pip install streamlit")

def launch_gradio():
    """Launch Gradio app"""
    print("\n🚀 Launching Gradio...")
    print("This will open your browser at http://localhost:7860")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run(
            [sys.executable, "app_gradio.py"],
            cwd=os.path.dirname(__file__)
        )
    except KeyboardInterrupt:
        print("\n\n✅ Gradio server stopped")
    except Exception as e:
        print(f"\n❌ Error launching Gradio: {e}")
        print("\nTry installing:")
        print("  pip install gradio")

def main():
    """Main launcher loop"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies first:")
        print("  pip install streamlit gradio")
        return
    
    print("\n✅ All dependencies installed!")
    
    while True:
        show_menu()
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            launch_streamlit()
            break
        elif choice == "2":
            launch_gradio()
            break
        elif choice == "3":
            show_system_info()
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
