from transformers import pipeline
generator = pipeline('text-generation', model='https://hf-mirror.com/EleutherAI/gpt-neo-125m')
generator("EleutherAI has", do_sample=True, min_length=20)