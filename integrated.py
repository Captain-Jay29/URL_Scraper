# File: integrated_flow.py

import requests
from urllib.parse import quote
from googlesearch import search
import re
import torch
import os
import html
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
import traceback
import os
from dotenv import load_dotenv

load_dotenv()

# Ensure NLTK tokenizer data is available
nltk.download('punkt', quiet=True)

###############################
# Configuration & Constants
###############################

# For scraping
API_KEY = os.getenv("Jina_API_KEY")
print(f'api-key for jina = ', API_KEY)

EXCLUDED_DOMAINS = ['youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com']

# Filenames
OUTPUT_FILENAME = "output.txt"
SUMMARY_FILENAME = "summaries_new.txt"

# For summarization
DEBUG = True
MAX_RETRIES = 2  # For GPU error recovery

###############################
# Jina Flow Functions (Scraping)
###############################

def get_urls(query, num_results=7):
    """
    Use googlesearch-python to fetch a list of URLs for the given query,
    excluding links from video streaming platforms.
    """
    raw_urls = list(search(query, num_results=num_results * 2))
    filtered_urls = []
    for url in raw_urls:
        if any(domain in url for domain in EXCLUDED_DOMAINS):
            continue
        filtered_urls.append(url)
        if len(filtered_urls) >= num_results:
            break
    return filtered_urls

def get_clean_content(url):
    """
    Fetch and return clean content for a given URL using the Jina Reader API.
    """
    encoded_url = quote(url, safe='')
    api_endpoint = f"https://r.jina.ai/{encoded_url}"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-Token-Budget": "200000",
        "X-Engine": "direct"
    }
    
    response = requests.get(api_endpoint, headers=headers, timeout=60)
    
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

def fetch_and_write_output(mode, query=None, url_list=None, num_results=5):
    """
    Depending on the mode, either fetch URLs via a query or use a provided URL list.
    Then scrape content from each URL and write results to OUTPUT_FILENAME.
    """
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        f.write("=== Search Results ===\n")
        
        # Determine URL list based on the user's choice
        if mode == "query":
            f.write(f"Query used: {query}\n")
            urls = get_urls(query, num_results=num_results)
        elif mode == "urls":
            urls = url_list
        else:
            raise ValueError("Invalid mode specified.")

        if not urls:
            f.write("No URLs found.\n")
            return
        
        f.write("\n=== Found URLs ===\n")
        f.write('\n'.join(urls) + '\n')
        
        f.write("\n=== Cleaned Content ===\n")
        for url in urls:
            try:
                content = get_clean_content(url)
                f.write(f"\nURL: {url}\n")
                f.write("-" * 50 + "\n")
                f.write(content + "\n")
                f.write("=" * 80 + "\n")
            except Exception as e:
                f.write(f"\nError processing {url}:\n{str(e)}\n")

###############################
# Summarization Flow Functions
###############################

def clean_content(text):
    """
    Remove HTML/Markdown tags and clean text for summarization.
    """
    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # Remove markdown images
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)    # Remove markdown links
    text = re.sub(r"#{1,6}\s*", "", text)          # Remove headers
    text = re.sub(r"\n+", ". ", text)              # Replace newlines with periods
    text = re.sub(r"\s+", " ", text)               # Remove extra spaces
    return text.strip()

def debug_print(*args):
    if DEBUG:
        print("[DEBUG]", *args)

def initialize_model():
    """
    Initialize and return the summarization pipeline.
    """
    print("\n=== Model Initialization ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    try:
        print("\nLoading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        debug_print(f"Moving model to {device}")
        model = model.to(device)
        summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=0 if device.startswith("cuda") else -1,
        )
        return summarizer
    except Exception as e:
        print(f"\n!!! Model loading failed: {str(e)}")
        print(traceback.format_exc())
        raise

def safe_summarize(summarizer, content):
    """
    Summarize content with error recovery and dynamic length checks.
    """
    for attempt in range(MAX_RETRIES):
        try:
            cleaned = clean_content(content)
            tokens = summarizer.tokenizer.encode(cleaned, truncation=False, add_special_tokens=False)
            if len(tokens) > 1024:
                debug_print(f"Truncating from {len(tokens)} tokens")
                cleaned = summarizer.tokenizer.decode(tokens[:1000], skip_special_tokens=True)
            max_len = min(150, len(tokens) // 2)
            min_len = max(10, max_len // 3)
            summary = summarizer(
                cleaned,
                max_length=max(30, max_len),
                min_length=min_len,
                do_sample=False,
                truncation=True
            )[0]['summary_text']
            return summary
        except RuntimeError as e:
            if "CUDA" in str(e) and attempt < MAX_RETRIES - 1:
                debug_print(f"GPU error, retrying ({attempt+1}/{MAX_RETRIES})")
                torch.cuda.empty_cache()
                continue
            raise
    return "Summary error: Maximum retries exceeded"

def process_cleaned_output(summarizer, input_filename=OUTPUT_FILENAME, output_filename=SUMMARY_FILENAME):
    """
    Process the scraped output and generate summaries.
    """
    print("\n=== Summarization Processing ===")
    print(f"Input file: {input_filename}")
    print(f"Output file: {output_filename}")
    
    try:
        if not os.path.exists(input_filename):
            raise FileNotFoundError(f"{input_filename} not found in {os.getcwd()}")
        
        with open(input_filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split the file into sections starting from the cleaned content
        try:
            cleaned_section = re.split(r"=== Cleaned Content ===", content)[1].strip()
        except IndexError:
            raise ValueError("Cleaned content section not found in the output file.")
        
        url_blocks = re.split(r"\n=+", cleaned_section)
        results = []
        for i, block in enumerate(url_blocks):
            block = block.strip()
            if not block:
                continue

            debug_print(f"\nProcessing block {i+1}:")
            try:
                url_match = re.search(r"URL:\s*(.+?)\n", block)
                if not url_match:
                    debug_print("Skipping block with no URL")
                    continue
                url = url_match.group(1).strip()
                # Remove the URL header and separator lines
                content_block = re.sub(r"^URL:.*\n-+\n", "", block, flags=re.MULTILINE)
                if not content_block.strip():
                    raise ValueError("Empty content after cleaning")
                
                summary = safe_summarize(summarizer, content_block)
                results.append(f"URL: {url}\n{'-'*50}\n{summary}\n{'='*80}")
            except Exception as e:
                error_msg = f"Error processing block {i+1}: {str(e)}"
                debug_print(error_msg)
                results.append(f"URL: {url}\n{'-'*50}\n{error_msg}\n{'='*80}")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(results))
        
        print(f"\nSummarization complete. {len(results)} summaries written to '{output_filename}'.")
    except Exception as e:
        print(f"\n!!! Critical error during summarization: {str(e)}")
        print(traceback.format_exc())
        raise

###############################
# Main Integrated Flow
###############################

def main():
    """
    Integrated flow:
    1. Ask user to either search via a query or provide a list of URLs.
    2. Scrape content from URLs and write to OUTPUT_FILENAME.
    3. Summarize the scraped content and write summaries to SUMMARY_FILENAME.
    """
    print("Welcome to the integrated scraping and summarization tool.")
    print("Choose an option:")
    print("1. Enter a search query (will use Google Search to fetch URLs)")
    print("2. Enter a list of URLs to scrape")
    
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        query = input("Enter your search query: ").strip()
        if not query:
            print("No query provided. Exiting.")
            return
        print("\nFetching URLs and scraping content based on your query...")
        fetch_and_write_output(mode="query", query=query, num_results=5)
    elif choice == "2":
        url_input = input("Enter URLs separated by commas: ").strip()
        if not url_input:
            print("No URLs provided. Exiting.")
            return
        # Process the input into a list of URLs (strip spaces)
        url_list = [url.strip() for url in url_input.split(",") if url.strip()]
        print("\nScraping content from the provided URLs...")
        fetch_and_write_output(mode="urls", url_list=url_list)
    else:
        print("Invalid option selected. Exiting.")
        return

    print(f"\nScraped data saved to '{OUTPUT_FILENAME}'.")
    print("\nInitializing summarization model...")
    summarizer = initialize_model()
    process_cleaned_output(summarizer)
    print(f"\nSummarized output saved to '{SUMMARY_FILENAME}'.\n")

if __name__ == "__main__":
    main()
