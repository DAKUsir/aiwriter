import openai  # For OpenAI integration
import gradio as gr
from openai import OpenAI

# Set your Nemotron API key
nemotron_api_key = "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"
nemotron_api_url = "https://integrate.api.nvidia.com/v1"  # Correct API base URL

# Initialize the client for Nemotron API
client = OpenAI(
    base_url=nemotron_api_url,
    api_key=nemotron_api_key
)

# Function to generate descriptions using Nemotron API
def generate_nemotron_description(product_name, features, audience):
    # Create the prompt based on the input
    input_text = (
        f"Write a highly detailed, professional, and attractive product description for a traditional craft item called '{product_name}'. "
        f"This product has the following features: {features}. "
        f"It is designed for {audience}. Highlight its cultural significance, craftsmanship, uniqueness, and appeal. "
        f"Use emotional and sensory-rich language to make it compelling. The description must be at least 300 words long and suitable for e-commerce or marketing."
    )

    try:
        # API call to the Nemotron API using the chat completion endpoint
        completion = client.chat.completions.create(
            model="nvidia/nemotron-4-340b-instruct",  # Adjust to your model name
            messages=[{"role": "user", "content": input_text}],
            temperature=0.7,
            top_p=0.9,
            max_tokens=250,
            stream=False  # Set to False to get the whole response at once
        )

        # Extract the generated description
        # Correctly access the message content
        generated_text = completion.choices[0].message.content
        return generated_text.strip()

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio interface for ease of use
interface = gr.Interface(
    fn=generate_nemotron_description,
    inputs=[
        gr.Textbox(label="Product Name", placeholder="e.g., Handwoven Silk Saree"),
        gr.Textbox(label="Features", placeholder="e.g., eco-friendly, handmade, intricate patterns"),
        gr.Textbox(label="Target Audience", placeholder="e.g., luxury buyers, art enthusiasts"),
    ],
    outputs=gr.Textbox(label="Detailed Product Description"),
    title="AI Product Description Generator (Nemotron)",
    description=(
        "Generate long, engaging, and highly detailed product descriptions using Nemotron's API. "
        "Perfect for traditional craft items, e-commerce listings, and marketing purposes."
    ),
    flagging_mode="never"
)

# Launch Gradio app
interface.launch(share=True)

