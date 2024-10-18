import streamlit as st
from PIL import Image
import os
print(os.getcwd())


class UiConfig:
    project_name = "Process Rationalization Genie"

    def setup():
        im = Image.open("/Users/tcml/Library/CloudStorage/OneDrive-Capco/Documents/Hackathon-usecase/process-rationalization/src/images/favicon.ico")

        st.set_page_config(page_title="Process genie", page_icon=im,layout="wide")

        t1, t2 = st.columns((0.25, 1))
        t1.title("")
        t1.image("/Users/tcml/Library/CloudStorage/OneDrive-Capco/Documents/Hackathon-usecase/process-rationalization/src/images/logo.png", width=100)
        t2.title("Process Rationalization Genie")
        t2.markdown("** A GenAI tool to identify Process Rationalization opportunities **")
