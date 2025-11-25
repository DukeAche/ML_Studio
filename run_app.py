#!/usr/bin/env python3
"""
Simple script to run the No-Code ML Studio application
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    try:
        # Change to the directory containing the app
        app_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(app_dir)
        
        # Run the Streamlit app
        print("🚀 Starting No-Code ML Studio...")
        print("📱 Opening in your default browser...")
        print("📝 If the browser doesn't open automatically, check the terminal for the URL")
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], 
                      check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting the application: {e}")
        print("💡 Make sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it:")
        print("   pip install streamlit")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()