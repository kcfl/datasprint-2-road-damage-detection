"""Interactive Web Interface for Road Damage Detection powered by Moonshot Kimi 3.1 Pro (NVIDIA NIM)."""

import os
import sys

# Ensure src/ is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
from road_damage.kimi_client import KimiVisionClient

# Initialize client
client = None
try:
    client = KimiVisionClient()
except Exception as e:
    print(f"Warning initializing KimiVisionClient: {e}")

DEFAULT_PROMPT = (
    "Examine this road surface image closely. Identify all forms of pavement distress "
    "(such as potholes, longitudinal cracks, transverse cracks, or alligator fatigue cracking). "
    "Provide a detailed breakdown including:\n"
    "1. Detected Damage Types\n"
    "2. Approximate Locations & Visual Indicators\n"
    "3. Severity Level (Low / Moderate / Critical)\n"
    "4. Recommended Maintenance or Repair Action"
)

PRESET_PROMPTS = {
    "🔍 Full Distress Assessment": DEFAULT_PROMPT,
    "🕳️ Potholes & Cracks Detection": (
        "Focus specifically on locating potholes and cracks in this image. List their approximate bounding areas "
        "and describe their depth and width characteristics."
    ),
    "⚠️ Safety & Severity Priority": (
        "Evaluate this pavement image from a traffic safety perspective. Assign a severity grade (1-5), "
        "estimate immediate hazard to vehicles, and recommend urgency for road crews."
    ),
}


def analyze_road_image(image, prompt, max_tokens):
    """Handler for streaming road damage analysis."""
    if image is None:
        yield "⚠️ Please upload or provide a road image to analyze."
        return

    if not prompt or not prompt.strip():
        prompt = DEFAULT_PROMPT

    global client
    if client is None:
        try:
            client = KimiVisionClient()
        except Exception as err:
            yield f"❌ Error initializing NVIDIA NIM client: {err}"
            return

    # Gradio passes filepath or numpy array depending on type; filepath string is supported directly
    partial_text = ""
    try:
        for chunk in client.analyze_stream(prompt=prompt, image_source=image, max_tokens=int(max_tokens)):
            partial_text += chunk
            yield partial_text
    except Exception as e:
        yield f"❌ Error during inference: {e}"


def create_ui():
    custom_css = """
    .gradio-container { max-width: 1200px !important; margin: auto; }
    .header-box { text-align: center; margin-bottom: 20px; }
    """

    with gr.Blocks(title="Road Damage AI — Kimi 3.1 Pro", css=custom_css, theme=gr.themes.Soft()) as demo:
        with gr.Row(elem_classes=["header-box"]):
            gr.Markdown(
                """
                # 🛣️ Road Damage Detection & Analysis
                ### Powered by **Moonshot AI Kimi 3.1 Pro** via NVIDIA NIM
                *Multimodal 3T-class reasoning model for automated pavement inspection, crack localization, and severity assessment.*
                """
            )

        with gr.Row():
            with gr.Column(scale=5):
                image_input = gr.Image(
                    label="Road / Pavement Image",
                    type="filepath",
                    sources=["upload", "clipboard"],
                )

                gr.Markdown("#### ⚡ Quick Analysis Templates")
                with gr.Row():
                    btn_full = gr.Button("🔍 Full Distress Assessment", size="sm")
                    btn_cracks = gr.Button("🕳️ Potholes & Cracks", size="sm")
                    btn_safety = gr.Button("⚠️ Safety Priority", size="sm")

                prompt_input = gr.Textbox(
                    label="Prompt / Inspection Instructions",
                    value=DEFAULT_PROMPT,
                    lines=5,
                    placeholder="Enter instructions for Kimi 3.1 Pro...",
                )

                with gr.Accordion("⚙️ Model Parameters", open=False):
                    max_tokens_slider = gr.Slider(
                        minimum=256,
                        maximum=8192,
                        value=2048,
                        step=256,
                        label="Max Tokens",
                    )
                    model_info = gr.Textbox(
                        label="Active Model",
                        value="moonshotai/kimi-k3.1-pro (NVIDIA NIM)",
                        interactive=False,
                    )

                with gr.Row():
                    submit_btn = gr.Button("🚀 Analyze Road Damage", variant="primary", size="lg")
                    clear_btn = gr.ClearButton([image_input, prompt_input], size="lg")

            with gr.Column(scale=7):
                output_text = gr.Markdown(
                    label="Kimi 3.1 Pro Analysis",
                    value="*Analysis output will stream here in real time...*",
                )

        # Wire template buttons
        btn_full.click(lambda: PRESET_PROMPTS["🔍 Full Distress Assessment"], outputs=prompt_input)
        btn_cracks.click(lambda: PRESET_PROMPTS["🕳️ Potholes & Cracks Detection"], outputs=prompt_input)
        btn_safety.click(lambda: PRESET_PROMPTS["⚠️ Safety & Severity Priority"], outputs=prompt_input)

        # Wire submission
        submit_btn.click(
            fn=analyze_road_image,
            inputs=[image_input, prompt_input, max_tokens_slider],
            outputs=output_text,
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
