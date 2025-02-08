

import argparse
import json
import logging
import os
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def initialize_generator(model_name: str = 'gpt2'):
    """
    Initialize the text generation pipeline using Hugging Face's model.
    """
    try:
        logging.info(f"Loading model '{model_name}'...")
        generator = pipeline('text-generation', model=model_name)
        logging.info("Model loaded successfully.")
        return generator
    except Exception as e:
        logging.error(f"Error initializing the model: {e}")
        return None


def get_user_input():
    """
    Prompt the user for details to create a custom prompt.
    """
    print("\n--- Create a New Prompt ---")
    purpose = input("What is the purpose of your prompt? (e.g., 'Write a blog post', 'Explain quantum physics'): ").strip()
    print("\nAnswer the following questions to refine your prompt:")
    target_audience = input("Who is the target audience? (e.g., 'students', 'professionals', 'general public'): ").strip()
    tone = input("What tone should the response have? (e.g., 'formal', 'casual', 'technical'): ").strip()
    length = input("How long should the response be? (e.g., 'short', 'medium', 'detailed'): ").strip()
    specific_details = input("Any specific details or keywords to include? (e.g., 'focus on AI ethics', 'use simple language'): ").strip()

    return {
        "purpose": purpose,
        "target_audience": target_audience,
        "tone": tone,
        "length": length,
        "specific_details": specific_details
    }


def generate_prompt(user_input: dict) -> str:
    """
    Generate a prompt string based on user input.
    """
    prompt = (
        f"Write a {user_input['length']} response for {user_input['target_audience']} "
        f"about {user_input['purpose']}. The tone should be {user_input['tone']}. "
        f"Additional details: {user_input['specific_details']}."
    )
    return prompt


def generate_response(generator, prompt: str, max_length: int = 200, num_return_sequences: int = 1):
    """
    Generate responses from the given prompt using the Hugging Face model.
    """
    try:
        responses = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
        return [resp.get('generated_text', '') for resp in responses]
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return None


def save_prompt(prompt: str, filename: str = "prompt.json"):
    """
    Save the generated prompt to a JSON file.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({"prompt": prompt}, f, ensure_ascii=False, indent=4)
        logging.info(f"Prompt saved to '{filename}'")
    except Exception as e:
        logging.error(f"Error saving prompt: {e}")


def load_prompt(filename: str = "prompt.json") -> str:
    """
    Load a prompt from a JSON file.
    """
    if not os.path.exists(filename):
        logging.error(f"File '{filename}' does not exist.")
        return None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("prompt", None)
    except Exception as e:
        logging.error(f"Error loading prompt: {e}")
        return None


def display_responses(responses):
    """
    Display the generated responses.
    """
    if responses:
        for idx, response in enumerate(responses, start=1):
            print(f"\n--- Generated Response {idx} ---\n{response}\n")
    else:
        print("No responses were generated.")


def generate_and_display_responses(generator, prompt: str, max_length: int, num_return_sequences: int):
    """
    Generate responses using the model and display them.
    """
    print("\nGenerating response(s)... Please wait.")
    responses = generate_response(generator, prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    display_responses(responses)


def main_menu(generator, max_length: int, num_return_sequences: int):
    """
    Display the main menu and handle user choices.
    """
    while True:
        print("\n=== Main Menu ===")
        print("1. Generate a new prompt")
        print("2. Load a saved prompt")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == '1':
            user_input = get_user_input()
            prompt = generate_prompt(user_input)
            print("\n--- Generated Prompt ---")
            print(prompt)
            generate_and_display_responses(generator, prompt, max_length, num_return_sequences)
            if input("Do you want to save this prompt? (yes/no): ").strip().lower() == "yes":
                filename = input("Enter filename to save prompt (default 'prompt.json'): ").strip() or "prompt.json"
                save_prompt(prompt, filename)
        elif choice == '2':
            filename = input("Enter filename to load prompt (default 'prompt.json'): ").strip() or "prompt.json"
            loaded_prompt = load_prompt(filename)
            if loaded_prompt:
                print("\n--- Loaded Prompt ---")
                print(loaded_prompt)
                generate_and_display_responses(generator, loaded_prompt, max_length, num_return_sequences)
            else:
                print("Failed to load the prompt. Please check the filename and try again.")
        elif choice == '3':
            print("Exiting... Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="AI Prompt Generator Bot")
    parser.add_argument(
        "--model", type=str, default="gpt2",
        help="Name of the Hugging Face model to use (default: 'gpt2')."
    )
    parser.add_argument(
        "--max_length", type=int, default=200,
        help="Maximum length of the generated response (default: 200)."
    )
    parser.add_argument(
        "--num_return_sequences", type=int, default=1,
        help="Number of responses to generate (default: 1)."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    generator = initialize_generator(args.model)
    if not generator:
        logging.error("Generator initialization failed. Exiting.")
        return

    print("Welcome to the AI Prompt Generator Bot!")
    print("This tool helps you create effective prompts and generates sample responses using a Hugging Face model.")
    main_menu(generator, args.max_length, args.num_return_sequences)


if __name__ == "__main__":
    main()
