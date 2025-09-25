import gradio as gr
from unsloth import FastLanguageModel
import torch

# Model yükle
print("Model yükleniyor...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/ika/yzlm/llm/Kubex_Lmm_Finetune/qwen-kubernetes-0.0.7/checkpoint-6000",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

def generate_response(message, history):
    # Chat template formatla
    prompt = f"""<|im_start|>system
Sen Kubernetes uzmanı bir asistansın. Detaylı ve yararlı cevaplar ver. Sorulan soru özelinde cevap ver.<|im_end|>
<|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
"""
    
    # Device'ı belirle ve input'ları gönder
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            repetition_penalty=1.1
        )
    
    # Full response'u decode et
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Sadece assistant kısmını al - en son assistant'tan sonrasını
    right,left = full_response.split("assistant")
    clean_answer = left
    
    # Kalan tag'leri ve gereksiz kısımları temizle
    clean_answer = clean_answer.replace("<|im_end|>", "")
    clean_answer = clean_answer.replace("<|im_start|>", "")
    
    # İlk satır boşsa kaldır
    lines = clean_answer.split('\n')
    while lines and lines[0].strip() == "":
        lines.pop(0)
    
    clean_answer = '\n'.join(lines).strip()
    
    return clean_answer

# Gradio interface
iface = gr.ChatInterface(
    fn=generate_response,
    title="🚢 Kubernetes Assistant",
    description="Kubernetes hakkında sorular sorun! Fine-tuned Qwen3-8B modeli kullanılıyor.",
    examples=[
        "Kubernetes nedir ve neden kullanılır?",
        "Pod nasıl oluşturulur?",
        "Deployment ve ReplicaSet arasındaki fark nedir?",
        "Service türleri nelerdir?",
        "ConfigMap ve Secret nasıl kullanılır?",
        "Ingress nedir ve nasıl yapılandırılır?"
    ]
)

if __name__ == "__main__":
    print("Gradio interface başlatılıyor...")
    iface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )