from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

# Load the model and tokenizer
device = "cuda"  # Ensure CUDA is available and the right device is selected
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)


PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

<s> [INST] 
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# hey [/INST] Hello there! It's nice to meet you. How can I help you today? If you have a question or need some assistance with something, feel free to ask and I'll do my best to provide you with accurate and helpful information. If your question is not clear or does not make sense, I'll let you know and try to help clarify it if possible. If I don't have the answer to your question, I won't provide false information – instead, I'll let you know that I don't know and try to direct you to other resources that might be able to help. Let me know what I can do for you!</s>


PARAMETERS = {
    "temperature": 0.9,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "top_k": 50,
    "truncate": 1000,
    "max_new_tokens": 1024,
    "seed": 42,
    "stop_sequences": ["</s>"],
}

# messages = []
def chat_with_model(user_input, history, memory_limit=4):
    

    if len(history) > memory_limit*2:
        history = history[-2*memory_limit:]
   
    if len(history) == 0:
        history.append({"role": "user", "content": PROMPT + user_input})
    else:
        history.append({"role": "user", "content": user_input})

    encodeds = tokenizer.apply_chat_template(history, return_tensors="pt")
    model_inputs = encodeds.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    response = tokenizer.batch_decode(generated_ids)

    # messages.append({"role": "assistant", "content": response[0]})
    print(response[0])
    return response[0]

def predict(message, history):
    query = chat_with_model(message, history=history)
    yield query
   

gr.ChatInterface(predict).queue().launch()

# Set up Gradio interface
# iface = gr.Interface(
#     fn=chat_with_model,
#     inputs=gr.inputs.Textbox(lines=2, placeholder="Type your message here..."),
#     outputs="text",
#     theme="default"
# )

# # Start the Gradio app
# iface.launch()