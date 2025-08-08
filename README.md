# AgriGemma-3n: On-Device Agricultural Intelligence

<div align="center">
<img width="256" height="256" alt="ChatGPT Image Aug 9, 2025, 04_26_09 AM" src="https://github.com/user-attachments/assets/178cb20d-6a2f-4401-a493-ca39df0bc348" />

</div>


<div align="center">

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/Nadhari/agrigemma-3n-6893b7900bca2c482381e8c7)
[![Demo](https://img.shields.io/badge/🚀-Live%20Demo-blue)](https://huggingface.co/spaces/Nadhari/agrigemma-3n-demo)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

## 🚀 Try It Now - No Installation Required!

**[→ Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Nadhari/agrigemma-3n-demo)**

Experience AgriGemma-3n instantly in your browser - upload crop images, ask questions, get expert agricultural advice!

## 📱 Run on Your Phone

Watch it in action (Text-only): [📹 Phone Demo Video](https://youtu.be/l7n7PV4ruhY)

### iOS & Android Instructions:

1. **Download [PocketPal AI app](https://github.com/a-ghorbani/pocketpal-ai)**
   - [📱 iOS App Store](https://apps.apple.com/us/app/pocketpal-ai/id6502579498)
   - [🤖 Google Play Store](https://play.google.com/store/apps/details?id=com.pocketpalai)

2. **Search for the model** in the app:
   ```
   Nadhari/AgriGemma-3n-E2B-it-gguf
   ```

3. **Download and start diagnosing** - chat with it for instant expert advice!

## 💻 Run on Desktop (MUltimodal)

Watch desktop demo: [📹 Ollama Demo Video](https://youtu.be/nerN_UOIznw)

### Quick Start with Ollama

```bash
ollama run hf.co/Nadhari/AgriGemma-3n-E2B-it-gguf:Q8_0
```

### Enhanced Experience with Ollamate

<img width="1101" height="764" alt="Screenshot 2025-08-07 at 5 56 31" src="https://github.com/user-attachments/assets/67952a61-4dd4-498b-a72c-9c02272fb3ab" />


For multimodal chat with **image and text support**, use [Ollamate](https://github.com/humangems/ollamate) - check their repository for detailed installation instructions.

---

## 🌾 About AgriGemma-3n

### The Challenge We Address

Every year, crop diseases destroy approximately 20-40% of global agricultural production, translating to economic losses exceeding $220 billion. For smallholder farmers in developing regions, these losses can mean the difference between prosperity and poverty. While agricultural experts exist, their reach is limited. For instance in Sub-Saharan Africa, the ratio of extension workers to farmers can be as low as 1:1000.

Traditional computer vision approaches to crop disease diagnosis, while technically impressive, fail to bridge the knowledge gap. They can identify diseases but cannot engage in the nuanced, conversational support farmers need: understanding symptoms, explaining disease progression, recommending context-appropriate treatments, and adapting advice to local conditions.

### Our Approach

AgriGemma-3n transforms crop disease diagnosis from a classification problem into a comprehensive agricultural advisory system. Built on Gemma 3n's efficient architecture, our models combine:

1. **Visual Understanding**: Fine-tuned on a subset of 137,000 carefully curated images spanning 60 disease categories across 16 major crops
2. **Domain Expertise**: Trained on a subset 1 million question-answer pairs covering diagnosis, prevention, and treatment strategies
3. **Conversational Intelligence**: Capable of multi-turn discussions that mirror consultations with agricultural experts
4. **On-Device Deployment**: Optimized to run on mobile devices with limited connectivity

## 🤗 Model Collection

Explore our complete model collection on Hugging Face:

**[→ AgriGemma-3n Collection](https://huggingface.co/collections/Nadhari/agrigemma-3n-6893b7900bca2c482381e8c7)**

### Available Models

- **[AgriGemma-3n-E2B-it](https://huggingface.co/Nadhari/AgriGemma-3n-E2B-it)**: Maximum performance variant for tablets and newer smartphones
- **[AgriGemma-3n-E4B-it](https://huggingface.co/Nadhari/AgriGemma-3n-E4B-it)**: Efficiency-optimized variant for older devices and edge deployment
- **GGUF Versions**: Quantized models for mobile and edge deployment

## 🌟 Key Features

- 🔬 **Accurate Disease Diagnosis**: Identifies crop diseases from images with high precision
- 💬 **Conversational Guidance**: Provides detailed explanations and treatment recommendations
- 📱 **Offline Capable**: Works without internet connection after initial download
- 🏃 **Fast Inference**: Optimized for real-time responses on mobile devices
- 🎯 **Context-Aware**: Adapts recommendations to local conditions and farming practices

## 🛠️ Technical Details

### Training

Using Unsloth, we implemented a comprehensive LoRA-based fine-tuning strategy:

<img width="1357" height="636" alt="Screenshot 2025-08-09 at 3 05 35" src="https://github.com/user-attachments/assets/f5009cc4-d451-4a90-85ff-4bf6e56d4d44" />


<img width="1357" height="636" alt="Screenshot 2025-08-09 at 3 05 47" src="https://github.com/user-attachments/assets/d4b7b855-a427-4045-a365-31591571e800" />

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r              = 32,
    lora_alpha     = 32,
    lora_dropout   = 0.05,
    bias           = "none",
    random_state   = 3407,
    target_modules = "all-linear",
)
```


This approach enables the model to learn domain-specific visual features critical for accurate diagnosis while maintaining conversational abilities.

### Dataset

<img width="563" height="450" alt="Screenshot 2025-08-09 at 4 18 48" src="https://github.com/user-attachments/assets/84a0a70e-1ad2-4fd8-a30f-e111b52edf82" />


We use **9000** samples of the [Crop Diseases Domain Multimodal Dataset](https://arxiv.org/pdf/2503.06973v1)
The whole dataset contains:
- **137,000 images** spanning 60 disease categories across 16 major crops
- **1 million Q&A pairs** covering diagnosis, prevention, and treatment
- Expert-validated annotations
- Balanced distribution across crop types and diseases

## 🚀 Quick Start for Developers

### Using Transformers

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

# Load model and processor
model = AutoModelForVision2Seq.from_pretrained("Nadhari/AgriGemma-3n-E2B-it")
processor = AutoProcessor.from_pretrained("Nadhari/AgriGemma-3n-E2B-it")

# Load and process image
image = Image.open("path/to/crop_disease_image.jpg")
prompt = "What disease is affecting this plant and how can I treat it?"

# Generate response
inputs = processor(images=image, text=prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```


##  Contributing

We welcome contributions! 


## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Crop Diseases Domain Multimodal Datset
```bibtex
@article{liu2025multimodal,
  title={A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis},
  author={Liu, X. and Zhou, J. and others},
  journal={arXiv preprint arXiv:2503.06973},
  year={2025}
}
```

## 📬 Contact

For questions, suggestions, or collaborations:
- Open an issue on GitHub
- Reach out on [Hugging Face](https://huggingface.co/Nadhari)

---

<div align="center">
Made with ❤️
</div>
