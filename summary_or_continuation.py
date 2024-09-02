from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def summarize_text(input_sentence):
    inputs = tokenizer(input_sentence, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarizer.generate(inputs["input_ids"], max_length=50, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_continuation(input_sentence):
    continuation = text_generator(input_sentence, max_length=50, num_return_sequences=1)[0]['generated_text']    
    return continuation


user_input = input("Enter the text you want to process:\n")

summarization_model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
summarizer = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)
print("\nSummary:", summarize_text(user_input))

generation_model_name = "gpt-3.5-turbo"
text_generator = pipeline("text-generation", model=generation_model_name)
print("\nContinuation:", generate_continuation(user_input))