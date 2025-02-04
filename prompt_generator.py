from transformers import pipeline

# Initialize the text generation pipeline using Hugging Face's GPT-2 model
generator = pipeline('text-generation', model='gpt2')

def get_user_input():
    """
    Get user input for the purpose of the prompt and additional details.
    """
    purpose = input("What is the purpose of your prompt? (e.g., 'Write a blog post', 'Explain quantum physics'): ")
    print("\nAnswer the following questions to refine your prompt:")
    
    # Questions to refine the prompt
    target_audience = input("Who is the target audience? (e.g., 'students', 'professionals', 'general public'): ")
    tone = input("What tone should the response have? (e.g., 'formal', 'casual', 'technical'): ")
    length = input("How long should the response be? (e.g., 'short', 'medium', 'detailed'): ")
    specific_details = input("Any specific details or keywords to include? (e.g., 'focus on AI ethics', 'use simple language'): ")
    
    return {
        "purpose": purpose,
        "target_audience": target_audience,
        "tone": tone,
        "length": length,
        "specific_details": specific_details
    }

def generate_prompt(user_input):
    """
    Generate a prompt based on user input.
    """
    prompt = (
        f"Write a {user_input['length']} response for {user_input['target_audience']} "
        f"about {user_input['purpose']}. The tone should be {user_input['tone']}. "
        f"Additional details: {user_input['specific_details']}."
    )
    return prompt

def generate_response(prompt):
    """
    Generate a response using Hugging Face's GPT-2 model.
    """
    response = generator(prompt, max_length=200, num_return_sequences=1)
    return response[0]['generated_text']

def main():
    print("Welcome to the AI Prompt Generator!")
    print("This tool will help you create effective prompts for AI models like ChatGPT.\n")
    
    # Get user input
    user_input = get_user_input()
    
    # Generate the prompt
    prompt = generate_prompt(user_input)
    print("\nGenerated Prompt:\n", prompt)
    
    # Generate a response using Hugging Face's GPT-2
    print("\nGenerating a sample response using Hugging Face's GPT-2...\n")
    response = generate_response(prompt)
    print("Generated Response:\n", response)

if __name__ == "__main__":
    main()