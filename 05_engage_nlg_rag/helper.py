import time
import openai
import numpy as np
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai.embeddings_utils import get_embedding
import faiss
from typing import List

class Helper():

    def __init__(self, api_key):
        openai.api_key = api_key
        self.api_key = api_key

    def nlg(self, dataset: pd.DataFrame, column_name: str):
        columns_info = dataset.dtypes
        categorical_features = dataset.select_dtypes(include=['object']).columns
        numerical_features = dataset.select_dtypes(include=['int64', 'float64']).columns      
        unique_values = {col: get_unique_values(dataset, col) for col in categorical_features}
        numerical_stats = dataset[numerical_features].describe()
        template = generate_template(dataset, columns_info, categorical_features, numerical_features, unique_values, numerical_stats)

        dataset[column_name] = dataset.apply(lambda row: populate_template(row, template), axis=1)

        return dataset

    def generate_embeddings(self, documents):
        embeddings = [get_embedding(doc, engine='text-embedding-3-small') for doc in documents]
        embeddings_np = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(embeddings_np)

        return index

    def generate_context(self, query: str, index, documents: List[str], k: int = 5, threshold: float = 0.5):
        query_embedding = get_embedding(query, engine="text-embedding-3-small")
        query_embeddings_np = np.array([query_embedding]).astype('float32')
        query_embeddings_np = query_embeddings_np.reshape(1, -1)
        
        # Increase k to ensure we get enough results
        D, I = index.search(query_embeddings_np, k * 2)
        
        # Filter results but ensure minimum number of results
        filtered_results = []
        for score, idx in zip(D[0], I[0]):
            if idx < len(documents):  # Ensure index is valid
                filtered_results.append(documents[idx])
                if len(filtered_results) >= k:  # Stop once we have enough results
                    break
        
        return filtered_results

    # Call model
    def call_model(self, messages, roles, contents):
        if isinstance(contents, list):
            # Batch processing
            responses = []
            for role, content in zip(roles, contents):
                temp_messages = messages.copy()
                temp_messages.append({"role": role, "content": content})
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=temp_messages,
                    temperature=0.5,
                )
                responses.append(response.choices[0].message.content)
            return responses
        else:
            # Single message processing
            messages.append({"role": roles, "content": contents})
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.5,
            )
            return response.choices[0].message.content

# Natural Language Generation Functions
def get_unique_values(dataset, column, limit=10):
    unique_values = dataset[column].unique().tolist()

    if len(unique_values) > limit:
        unique_values = unique_values[:limit] + ['...']
    return unique_values

def generate_template(dataset, columns_info, categorical_features, numerical_features, unique_values, numerical_stats):
    # Construct a summary of the dataframe's structure
    column_summary = "Column Names and Data Types:\n"
    for column, dtype in columns_info.items():
        column_summary += f"{column}: {dtype}\n"

    # Unique values for categorical features
    unique_values_str = "\nUnique Values for Categorical Features:\n"
    for col, uniques in unique_values.items():
        unique_values_str += f"{col}: {uniques}\n"

    # Descriptive statistics for numerical features
    numerical_stats_str = "\nDescriptive Statistics for Numerical Features:\n"
    for col in numerical_features:
        numerical_stats_str += f"{col}: \n"
        for stat_name, value in numerical_stats[col].items():
            numerical_stats_str += f"   {stat_name}: {value}\n"

    # Define the system prompt
    system_prompt = """
    You are an intelligent assistant that creates descriptive templates for transforming dataframe rows into coherent paragraphs.
    Analyze the provided dataframe structure and generate a template sentence that includes placeholders for each column.
    Ensure the template is contextually relevant and maintains grammatical correctness.
    """

    # Define the user prompt
    user_prompt = f"""
    Analyze the following dataframe structure and create a descriptive template with placeholders for each column.

    <column_summary>
    {column_summary}
    </column_summary>

    <unique_values>
    {unique_values_str}
    </unique_values>

    <numerical_stats>
    {numerical_stats_str}
    </numerical_stats>

    Use the exact column names from the column_summary in generating the variable names in the template,
    as they will be populated with the actual values in the dataset.

    Example Template about a Spotify dataset:
    "{{artist}} gained {{streams}} streams in the song '{{song}}' that was a hit in {{date}}."

    Output only the template without any explanation or introduction.
    The template's variables will be dynamically replaced so make sure they're formatted properly
    """

    # Generate the template (with retries)
    retries = 3
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            template = response['choices'][0]['message']['content'].strip()
            return template
        except Exception as e:
            print(f"Error generating template (Attempt {attempt + 1}/{retries}): {e}")
            time.sleep(2)  # Wait before retrying

    return None

def populate_template(row, template):
    # Convert row to dictionary and replace NaN with 'N/A'
    row_dict = row.to_dict()
    for key, value in row_dict.items():
      if pd.isna(value):
        row_dict[key] = 'N/A'

    # Replace placeholders in the template with actual values
    populated_template = template.format(**row_dict)
    return populated_template