A URL Scraper based on custom query. Exctracting content through Jina AI and summarizing using the BART summarizer on HuggingFace.

# Project Setup and Run Guide

This guide provides step-by-step instructions on how to set up and run the project.

---

## **1. Clone the Repository**
```bash
git clone git@github.com:Captain-Jay29/URL_Scraper.git
```

---

## **2. Set Up Environment Variables**
- An `example.env` file is provided with placeholders for your API key.
- Rename the `example.env` file to `.env`:
  ```bash
  mv example.env .env
  ```
- Open the `.env` file and add your API key:
  ```env
  API_KEY=your_actual_api_key_here
  ```

---

## **3. Install Dependencies**
- Use the provided `requirements.txt` to install all necessary Python packages:
  ```bash
  pip install -r requirements.txt
  ```

---

## **4. Run the Project**
- Execute the main Python script:
  ```bash
  python integrated.py
  ```

---

## **5. Notes**
- Ensure that your `.env` file is never pushed to version control.
- The `dotenv` package is used to load environment variables securely from the `.env` file.
- All dependencies required to run the project are listed in the `requirements.txt`.

---

You're all set! The project should now be running smoothly. Let me know if you face any issues.

