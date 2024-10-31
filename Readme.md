# Website Scraper and PDF Chunker/Vectorizer for Pinecone DB

This Python repository contains a set of scripts that allow you to scrape a website, clean the data, organize it, chunk it, and then vectorize it. The resulting vectors can be used for a variety of machine learning tasks, such as similarity search or clustering. Recently, a script was added to consume PDFs and add them to the training data as well.

## Files
THESE FUNCTIONS CONSUME THE FILES THEY PROCESS (only in the websites and pdfs directories)

- `cleaner.py`: This script downloads a website using wget, reads and cleans the HTML files using Beautiful Soup, and saves the resulting text files in a specified directory.
- `chunker.py`: This script splits the text files into smaller chunks using a recursive character-based text splitter. The resulting chunks are saved in a JSONL file.
- `vectorizor.py`: This script loads the JSONL file, creates embeddings using OpenAI's text-embedding-ada-002 model, and indexes the embeddings using Pinecone. This script requires API keys to run.
- `pdf-muncher.py`: This script processes every PDF in the pdf-docs folder and adds the content to the vectorized train.json.

## Requirements

- Python 3.x
- OpenAI API key
- Pinecone API key
- `bs4` Python library
- `jsonlines` Python library
- `tqdm` Python library
- `tiktoken` Python library
- `pinecone-client` Python library

## Usage

1. Clone the repository and navigate to the project directory.
2. Install the required Python libraries using `pip install -r requirements.txt`.
3. Set up your OpenAI and Pinecone API keys.
4. Download the website using the wget command:
  `wget -r -np -nd -A.html,.txt,.tmp -P websites https://www.linkedin.com/in/sean-stobo/`
5. Run `python cleaner.py` to download and clean the website data. This will break down the directory structure into a list of HTML documents.
6. Run `python chunker.py` to split the text files into smaller chunks. This outputs train.json in the root directory.
7. Run `python pdf-muncher.py` to convert the contents of the '/pdfs/' folder to a serialized train.jsonl file in the root directory.
8. Run `python vectorizor.py` to create embeddings and index them using Pinecone. This will vectorize train.json.

Note: Before running `vectorizor.py`, make sure to set up a Pinecone database with 1536 dimensions.

## Visual Guide

1. Choose a site to scrape.
  ![Step 1](data/pdf-1.png)
2. Observe the 'website' folder filling up with files.
  ![Step 2](data/pdf-2.png)
3. Run the cleaner script.
  ![Step 3](data/pdf-3.png)
4. Files are normalized and cleaned up.
  ![Step 4](data/pdf-4.png)
5. Run the chunker script to chunk and vectorize the website files.
  ![Step 5](data/pdf-5.png)
6. Run the PDF muncher script to process the PDFs in the pdfs folder.
  ![Step 6](data/pdf-muncher.png)
7. Verify that the vectorized training data contains the DnD content.
  ![Step 7](data/pdf-6.png)
8. Run the vectorizor script to update the Pinecone DB.