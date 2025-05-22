import streamlit as st
import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import your frontend UI
from frontend.ui import main

if __name__ == "__main__":
    main() 