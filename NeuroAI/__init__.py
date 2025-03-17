from langgraph.graph import StateGraph, END
import pandas as pd
from typing import TypedDict, Optional
from langchain.prompts import PromptTemplate
from groq import Groq
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import time
import regex as re
import json

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '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')
SERPER_API_KEY = 'edf28dbbb85930e14c617ad0eb0479799de050c1'
client = Groq(api_key=GROQ_API_KEY)

# Define the state structure
class DatasetState(TypedDict):
    paper_text: str
    dataset_name: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    year: Optional[str]
    access_type: Optional[str]
    institution: Optional[str]
    country: Optional[str]
    modality: Optional[str]
    subject: Optional[str]
    slice_scan_no: Optional[str]
    format: Optional[str]
    segmentation_mask: Optional[str]
    disease: Optional[str]

# Define the extraction prompt
EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
Extract the following information from the provided neuroradiology dataset paper text.
Your response should be concise and structured. If information is not found, return 'Not specified'.

Now, extract the following details from the given text:

1. Dataset Name - The official name of the dataset.
2. DOI - The Digital Object Identifier (DOI) if available.
3. URL - Link to access the dataset (if provided).
4. Year of Release - The year the dataset was published.
5. Access Type - Whether the dataset is open-access or restricted.
6. Institution - The university, research lab, or company that created the dataset.
7. Country - The country of the institution.
8. Modality - The imaging type (e.g., MRI, CT, X-ray).
9. Number of Subjects - The number of people or cases in the dataset.
10. Number of Slices/Scans - How many images/scans are available.
11. Format - The file format (e.g., DICOM, NIfTI).
12. Segmentation Mask - Whether segmentation masks are included (Yes/No).
13. Disease - The main disease(s) studied in the dataset.

Text to Analyze:
{text}

Output Format:
- Dataset Name: [answer]
- DOI: [answer]
- URL: [answer]
...

Final Instructions for AI:
- Extract information only from the provided text (do not assume details).
- If a field is missing, explain why instead of just saying `"Not specified"`.
- Verify that numerical values (year, subject count) are accurate.

"""

)


# Function to fetch and clean text from a webpage or PDF
def fetch_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Check if the URL is a PDF
        if url.lower().endswith(".pdf"):
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Read PDF content
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Extract text from all pages
            text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

        else:
            # Fetch webpage content
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unnecessary elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()

            # Extract text
            text = " ".join(soup.stripped_strings)

        return text

    except Exception as e:
        return f"Error fetching text from {url}: {str(e)}"


# Function to split text into smart chunks without cutting mid-sentence
def split_text_smart(text, max_tokens=5000):
    """
    Splits text into chunks of max_tokens, ensuring we don't cut mid-sentence.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split by punctuation (preserving sentence meaning)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())  # Estimate token count

        if current_length + sentence_length > max_tokens:
            # If adding this sentence exceeds the token limit, start a new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            # Otherwise, add to the current chunk
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))  # Add any remaining text

    return chunks


# Node to extract information using Groq
def extract_info(state: DatasetState) -> DatasetState:
    prompt = EXTRACTION_PROMPT.format(text=state["paper_text"])
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are an expert in extracting structured data from scientific texts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5000
        )
        result = response.choices[0].message.content
    except Exception as e:
        result = "\n".join([f"{k}: Not specified" for k in DatasetState.__annotations__ if k != "paper_text"])

    lines = result.split("\n")
    field_mapping = {
        "year_of_release": "year",
        "number_of_subjects": "subject",
        "number_of_slices/scans": "slice_scan_no"
    }

    for line in lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            key = key.strip("- ").lower().replace(" ", "_")
            if key in field_mapping:
                state[field_mapping[key]] = value.strip()
            elif key in DatasetState.__annotations__:
                state[key] = value.strip()

    return state

# Node to format and validate output
def format_output(state: DatasetState) -> DatasetState:
    fields = [
        "dataset_name", "doi", "url", "year", "access_type", "institution",
        "country", "modality", "subject", "slice_scan_no", "format",
        "segmentation_mask", "disease"
    ]
    for field in fields:
        if field not in state or state[field] is None:
            state[field] = "Not specified"
    return state


# Build the graph
workflow = StateGraph(DatasetState)
workflow.add_node("extract_info", extract_info)
workflow.add_node("format_output", format_output)
workflow.add_edge("extract_info", "format_output")
workflow.add_edge("format_output", END)
workflow.set_entry_point("extract_info")
app = workflow.compile()

# Function to ensemble results from multiple chunks
def ensemble_results(chunk_results):
    ensembled = {}
    fields = [k for k in DatasetState.__annotations__ if k != "paper_text"]
    for field in fields:
        values = [result.get(field, "Not specified") for result in chunk_results]
        for value in values:
            if value != "Not specified":
                ensembled[field] = value
                break
        else:
            ensembled[field] = "Not specified"
    return ensembled

# Function to analyze paper with chunking
def analyze_paper(paper_text: str) -> dict:
    chunks = split_text_smart(paper_text, max_tokens=5000)
    chunk_results = []
    for chunk in chunks:
        initial_state = {"paper_text": chunk}
        result = app.invoke(initial_state)
        chunk_results.append({k: v for k, v in result.items() if k != "paper_text"})
    return ensemble_results(chunk_results)

# Search function using Serper API
def serper_search(query: str, api_key: str, num_results: int = 10, page: int = 1) -> Optional[pd.DataFrame]:
    url = "https://google.serper.dev/search"
    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}
    payload = {"q": query, "num": num_results, "page": page}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json()
        organic_results = results.get('organic', [])
        processed = [
            {'Position': item.get('position'), 'Title': item.get('title'), 'URL': item.get('link')}
            for item in organic_results
        ]
        return pd.DataFrame(processed)
    except Exception as e:
        print(f"Search request failed: {str(e)}")
        return None

# Paginated search to collect results
def paginated_search(query: str, api_key: str, max_pages: int = 3) -> pd.DataFrame:
    all_results = pd.DataFrame()
    page = 1
    num_per_page = 10

    while page <= max_pages:
        print(f"Searching page {page}...")
        results_df = serper_search(query, api_key, num_per_page, page)
        if results_df is None or results_df.empty:
            print("No more results found or error occurred")
            break
        all_results = pd.concat([all_results, results_df], ignore_index=True)
        page += 1
        time.sleep(1)
    return all_results

# Process search results and extract information (use url)
def process_search_results(search_results: pd.DataFrame, max_urls: int = 5) -> pd.DataFrame:
    urls = search_results['URL'].head(max_urls).tolist()
    texts = []
    for idx, url in enumerate(urls):
        print(f"Processing URL {idx+1}/{len(urls)}: {url}")
        text = fetch_text_from_url(url)
        texts.append(text)
        time.sleep(2)

    results = [analyze_paper(text) for text in texts if text and not text.startswith("Error fetching URL")]
    df = pd.DataFrame(results)
    df['url'] = [url for url, text in zip(urls, texts) if text and not text.startswith("Error fetching URL")]
    return df

# Function to update the dataset with new deduplication logic
def update_dataset(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure consistent columns
    columns = [
        "dataset_name", "doi", "url", "year", "access_type", "institution",
        "country", "modality", "subject", "slice_scan_no", "format",
        "segmentation_mask", "disease"
    ]
    existing_df = existing_df.reindex(columns=columns, fill_value="Not specified")
    new_df = new_df.reindex(columns=columns, fill_value="Not specified")

    # Step 1: Deduplicate within new_df
    def is_duplicate_within(df):
        seen_keys = set()  # For dataset_name, doi, url
        seen_details = []  # For year, modality, slice_scan_no, subject comparison
        duplicates = []

        for idx, row in df.iterrows():
            # Primary key: dataset_name, doi, url
            primary_key = (row['dataset_name'], row['url'], row['doi'])
            # Secondary key: year, modality, slice_scan_no, subject
            secondary_key = (row['year'], row['modality'], row['slice_scan_no'], row['subject'])

            # Check primary key (dataset_name, doi, url)
            is_duplicate = any(k in seen_keys for k in primary_key if k != "Not specified")

            # If no primary match, check secondary key
            if not is_duplicate:
                match_count = 0
                for prev_key in seen_details:
                    # Count how many of the 4 fields match
                    matches = sum(1 for a, b in zip(secondary_key, prev_key) if a == b and a != "Not specified")
                    if matches >= 3:  # At least 3 out of 4 match
                        is_duplicate = True
                        break
                if not is_duplicate:
                    seen_details.append(secondary_key)

            duplicates.append(is_duplicate)
            if not is_duplicate:
                seen_keys.update(k for k in primary_key if k != "Not specified")

        return pd.Series(duplicates)

    new_duplicates = is_duplicate_within(new_df)
    new_df_cleaned = new_df[~new_duplicates].reset_index(drop=True)
    print(f"New data after internal deduplication: {len(new_df_cleaned)} rows (from {len(new_df)})")

    # Step 2: Deduplicate between new_df_cleaned and existing_df
    combined_df = pd.concat([existing_df, new_df_cleaned], ignore_index=True)
    print(f"Combined dataframe size: {len(combined_df)} rows")

    def is_duplicate_combined(df):
        seen_keys = set()  # For dataset_name, doi, url
        seen_details = []  # For year, modality, slice_scan_no, subject comparison
        duplicates = []

        for idx, row in df.iterrows():
            # Primary key: dataset_name, doi, url
            primary_key = (row['dataset_name'], row['url'], row['doi'])
            # Secondary key: year, modality, slice_scan_no, subject
            secondary_key = (row['year'], row['modality'], row['slice_scan_no'], row['subject'])

            # Check primary key (dataset_name, doi, url)
            is_duplicate = any(k in seen_keys for k in primary_key if k != "Not specified")

            # If no primary match, check secondary key
            if not is_duplicate:
                match_count = 0
                for prev_key in seen_details:
                    # Count how many of the 4 fields match
                    matches = sum(1 for a, b in zip(secondary_key, prev_key) if a == b and a != "Not specified")
                    if matches >= 3:  # At least 3 out of 4 match
                        is_duplicate = True
                        break
                if not is_duplicate:
                    seen_details.append(secondary_key)

            duplicates.append(is_duplicate)
            if not is_duplicate:
                seen_keys.update(k for k in primary_key if k != "Not specified")

        return pd.Series(duplicates)

    all_duplicates = is_duplicate_combined(combined_df)
    cleaned_df = combined_df[~all_duplicates].reset_index(drop=True)
    print(f"Final cleaned dataframe size: {len(cleaned_df)} rows")
    return cleaned_df

def search_dataset(df: pd.DataFrame, modality: str, disease: str, segmentation: str, access_type: str) -> dict:
    # Select only relevant columns and convert to string more efficiently
    relevant_columns = ['dataset_name', 'url', 'modality', 'disease', 'segmentation_mask', 'access_type']
    filtered_df = df[relevant_columns].head(20)  # Limit to 20 datasets for context

    # Create a more concise dataset representation
    dataset_entries = []
    for _, row in filtered_df.iterrows():
        entry = f"Dataset: {row['dataset_name']} | Modality: {row['modality']} | Disease: {row['disease']} | "
        entry += f"Segmentation: {row['segmentation_mask']} | Access: {row['access_type']} | URL: {row['url']}"
        dataset_entries.append(entry)

    dataset_context = "\n".join(dataset_entries)

    search_prompt = f"""
    You are a medical imaging dataset expert. Your task is to find datasets matching specific criteria.

    USER CRITERIA:
    - Modality: {modality}
    - Disease: {disease}
    - Segmentation Required: {segmentation}
    - Access Type: {access_type}

    AVAILABLE DATASETS:
    {dataset_context}

    INSTRUCTIONS:
    1. Find exact and partial matches based on the criteria
    2. Return ONLY a valid JSON object with this exact structure:
    {{
        "matches": [
            {{
                "dataset_name": "exact name from list",
                "url": "exact url from list",
                "relevance": "one sentence explanation"
            }}
        ],
        "alternatives": [
            {{
                "dataset_name": "exact name from list",
                "url": "exact url from list",
                "explanation": "one sentence explanation"
            }}
        ]
    }}

    IMPORTANT:
    - Use exact dataset names and URLs from the provided list
    - Include only valid JSON - no additional text
    - If no matches found, return empty arrays
    """

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise JSON-generating assistant. Always return valid JSON matching the requested structure."
                },
                {"role": "user", "content": search_prompt}
            ],
            temperature= 0,  # Reduced temperature for more consistent output
            max_tokens=1000
        )

        # Parse and display results
        result = response.choices[0].message.content.strip()

        try:
            # Remove any potential markdown code block markers
            result = result.replace("```json", "").replace("```", "").strip()
            parsed_results = json.loads(result)

            # Validate the response structure
            if not isinstance(parsed_results.get("matches"), list) or not isinstance(parsed_results.get("alternatives"), list):
                raise json.JSONDecodeError("Invalid response structure", result, 0)

            # Display matches
            if parsed_results["matches"]:
                print("\nBest Matching Datasets:")
                for match in parsed_results["matches"]:
                    print(f"\n- {match['dataset_name']}")
                    print(f"  URL: {match['url']}")
                    print(f"  Relevance: {match['relevance']}")
            else:
                print("\nNo exact matches found.")

            # Display alternatives
            if parsed_results["alternatives"]:
                print("\nAlternative Suggestions:")
                for alt in parsed_results["alternatives"]:
                    print(f"\n- {alt['dataset_name']}")
                    print(f"  URL: {alt['url']}")
                    print(f"  Note: {alt['explanation']}")

            return parsed_results

        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response. Using basic matching...")
            print(f"Raw response: {result}")  # For debugging
            return perform_basic_matching(filtered_df, modality, disease, segmentation, access_type)

    except Exception as e:
        print(f"Error querying LLM: {str(e)}")
        return perform_basic_matching(filtered_df, modality, disease, segmentation, access_type)

def perform_basic_matching(df, modality, disease, segmentation, access_type):
    """Fallback function for basic matching when LLM fails"""
    matches = []
    alternatives = []

    for _, row in df.iterrows():
        # Check for exact matches
        if (modality.lower() in str(row['modality']).lower() and
            disease.lower() in str(row['disease']).lower() and
            segmentation.lower() in str(row['segmentation_mask']).lower() and
            access_type.lower() in str(row['access_type']).lower()):
            matches.append({
                "dataset_name": row['dataset_name'],
                "url": row['url'],
                "relevance": "Matches all criteria"
            })
        # Check for partial matches
        elif (modality.lower() in str(row['modality']).lower() or
              disease.lower() in str(row['disease']).lower()):
            alternatives.append({
                "dataset_name": row['dataset_name'],
                "url": row['url'],
                "explanation": "Partial match on modality or disease"
            })

    return {"matches": matches, "alternatives": alternatives}

# Main execution: Update dataset first
print('Welcome to NeuroAI ...')
print('Updating spinal imaging dataset...')

# Load or initialize the dataset (assuming spinal sheet for this prototype)
dataset_file = 'dataset.xlsx'
sheet_name = 'spinal'
if os.path.exists(dataset_file):
    existing_df = pd.read_excel(dataset_file, sheet_name=sheet_name)
    print(f"Loaded {len(existing_df)} rows from existing dataset")
else:
    existing_df = pd.DataFrame(columns=[
        "dataset_name", "doi", "url", "year", "access_type", "institution",
        "country", "modality", "subject", "slice_scan_no", "format",
        "segmentation_mask", "disease"
    ])
    print("No existing dataset found, initializing empty dataframe")

# Update dataset with new search
SEARCH_QUERY = "spinal imaging dataset"
search_results = paginated_search(SEARCH_QUERY, SERPER_API_KEY, max_pages=3)
if not search_results.empty:
    new_df = process_search_results(search_results, max_urls=5)
    print(f"Found {len(new_df)} new rows from search")
    updated_df = update_dataset(existing_df, new_df)

    # Preserve other sheets and update only 'spinal'
    if os.path.exists(dataset_file):
        excel_book = pd.read_excel(dataset_file, sheet_name=None)  # Load all sheets
        excel_book[sheet_name] = updated_df  # Update spinal sheet
        with pd.ExcelWriter(dataset_file, engine='openpyxl', mode='w') as writer:
            for sheet, df in excel_book.items():
                df.to_excel(writer, sheet_name=sheet, index=False)
    else:
        with pd.ExcelWriter(dataset_file, engine='openpyxl', mode='w') as writer:
            updated_df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Dataset updated and saved to {dataset_file}, sheet: {sheet_name}")
else:
    updated_df = existing_df
    print("No new datasets found, using existing dataset.")
