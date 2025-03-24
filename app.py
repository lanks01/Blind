import streamlit as st
from PIL import Image
import base64

# Load and set background image (Spot-style)
def set_background(jpg_file):
    with open(jpg_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    .title-text {{
        font-size: 52px;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 6px black;
        text-align: center;
    }}
    .section-title {{
        font-size: 28px;
        font-weight: 700;
        color: white;
        text-shadow: 1px 1px 4px black;
        margin-top: 30px;
    }}
    .section-content {{
        font-size: 18px;
        color: #f0f0f0;
        text-shadow: 1px 1px 2px black;
        text-align: justify;
    }}
    .img-block {{
        margin-bottom: 25px;
        border: 2px solid #ffffff22;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 0 12px #00000055;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set the background (updated to .jpg)
set_background("image.jpg")

# Website content starts here
st.markdown('<div class="title-text">🔍 Blind Spot Detection System (Adverse Weather)</div>', unsafe_allow_html=True)
st.markdown('[👉 View GitHub Repository](https://github.com/ArnavMadavaram/BlindSpotDetector/tree/main)', unsafe_allow_html=True)

# Four interactive sections
options = st.columns(4)
with options[0]:
    carla_btn = st.button("CARLA")
with options[1]:
    sim_btn = st.button("Simulation")
with options[2]:
    process_btn = st.button("Data Processing")
with options[3]:
    final_btn = st.button("Final Output")

# === CARLA ===
if carla_btn:
    st.markdown('<div class="section-title">CARLA Simulation Environment</div>', unsafe_allow_html=True)
    st.image("CARLA.png", use_container_width=True, caption="CARLA Environment in Action")
    st.markdown(
        '<div class="section-content">CARLA is a high-fidelity open-source simulator for autonomous driving research. We used it to emulate urban scenarios under foggy and rainy conditions. The simulated environment allowed us to position the ego vehicle and various blind spot obstacles to create realistic data for training and evaluation.</div>',
        unsafe_allow_html=True,
    )

# === SIMULATION ===
if sim_btn:
    st.markdown('<div class="section-title">Multi-Sensor Simulation & Weather Effects</div>', unsafe_allow_html=True)
    st.image("Simulation.png", use_container_width=True, caption="Heavy Fog & Rain in CARLA")
    st.markdown(
        '<div class="section-content">We simulated multiple sensors like LiDAR, RADAR, RGB Camera, SWIR, and Depth Sensor using CARLA. Fog and rain were synthesized using weather APIs within the simulator, drastically reducing visibility. Sensor fusion became essential under these degraded conditions. Random noise and distortion were added to simulate real-world sensor interference. The sensors were mounted at realistic positions (side-view mirrors, rear bumper) to capture accurate blind spot data.</div>',
        unsafe_allow_html=True,
    )

# === DATA PROCESSING ===
if process_btn:
    st.markdown('<div class="section-title">Sensor Fusion & Data Preprocessing</div>', unsafe_allow_html=True)
    st.image("Data Processing.png", use_container_width=True, caption="Extracted Tensor Data from Simulated Sensors")
    st.markdown(
        '<div class="section-content">We extracted 6-channel tensors from the sensors, combining RGB, Depth, Semantic Segmentation, and Radar. Each fused tensor was normalized and stored as NumPy arrays. These were used as inputs to our deep learning model. A sample tensor contained values like RGB: (0.75, 0.75, 0.78), Depth: 0.00, Radar: 0.00, Segmentation ID: 3. This fusion helped the model make sense of the environment under visually obstructed conditions.</div>',
        unsafe_allow_html=True,
    )

# === FINAL OUTPUT ===
if final_btn:
    st.markdown('<div class="section-title">Model Output & Detection Visualization</div>', unsafe_allow_html=True)
    st.image("Final Output.png", use_container_width=True, caption="AI Prediction Results in Adverse Weather")
    st.markdown(
        '<div class="section-content">Our deep learning model processes the fused tensors and predicts the presence of objects in blind spots. If the prediction probability exceeds 0.5, the system classifies it as "Object Detected". Otherwise, it flags it as "No Object Detected". Output Example:<br><br><code>=== fused_285377 ===<br>Prediction Probability: 0.61<br>Decision: Object Detected</code><br><br>The model achieved reliable performance even under heavy fog, proving the robustness of our fusion approach.</div>',
        unsafe_allow_html=True,
    )
