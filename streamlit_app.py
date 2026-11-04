"""Streamlit Community Cloud entry point.

The desk reads the cached artifacts in this repository. It does not download
the public credit-card file. Point Cloud at this file, Python 3.12.
"""

from app.streamlit_app import main

main()
