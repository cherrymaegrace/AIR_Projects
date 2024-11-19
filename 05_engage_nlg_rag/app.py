import os
import openai
import numpy as np
import pandas as pd
import json
from helper import Helper
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

SYSTEM_PROMPT = '''
Role:
You are a marketing assistant specializing in personalized email campaigns. Your task is to generate effective and engaging email content tailored to specific customer profiles and behaviors.

Instruction:
Create personalized email drafts that cater to various marketing scenarios, including customer onboarding, reactivation, upselling, seasonal promotions, and feedback requests. Ensure the email includes a subject line and main content. Use customer segmentation data and purchase behavior provided as input to craft relevant and engaging messages.

Context:
The application is designed for generating targeted email campaigns based on customer segmentation data (e.g., demographics, purchase history, engagement level) and recent purchase patterns. The goal is to drive engagement and conversions through highly personalized communication.

Constraints:

Keep the email concise, professional, and engaging.
Use a tone appropriate to the scenario (e.g., friendly for onboarding, urgent for promotions).
Subject lines must be under 60 characters.
Main content should not exceed 150 words.
Ensure proper grammar and punctuation.
Examples:

Scenario: Onboarding (Welcome Email)

Subject: "Welcome to [Brand Name] – Let’s Get Started!"

Main Content:
"Hi [First Name],
Welcome to [Brand Name]! We're thrilled to have you join our community of [specific customer type, e.g., fitness enthusiasts].

As a Premium Member, you now have access to [list a few perks or features].

Ready to get started? Click [here] to personalize your experience.

Best,
The [Brand Name] Team"

Scenario: Reactivation (Lapsed Customer)

Subject: "We Miss You, [First Name]!"

Main Content:
"Hi [First Name],
It’s been a while, and we’d love to see you back at [Brand Name]! As a valued customer, we’ve got something special for you:

🎁 Exclusive Offer: [Discount/Offer Details]

Don’t miss out—this offer is valid until [specific date].

Looking forward to having you back!

Cheers,
The [Brand Name] Team"

Scenario: Upsell (Complementary Products)

Subject: "Complete Your Look, [First Name]!"

Main Content:
"Hi [First Name],
You recently purchased [Product Name], and we think you'll love these, too:

[Product 1] – [Short description]
[Product 2] – [Short description]
Bundle up and save! Get [specific discount] when you add these items to your cart.

Shop now and take your [specific use case, e.g., workouts, style] to the next level!

Best,
The [Brand Name] Team"

Scenario: Seasonal Promotion (Holiday Campaign)

Subject: "Holiday Deals Just for You, [First Name]!"

Main Content:
"Hi [First Name],
The holidays are here, and so are our best deals of the year! 🎄

🎁 Special Offer: Save up to [X]% on [specific products or categories].

Act fast—these deals are only available until [specific date].

Celebrate the season with [Brand Name] and make your holidays extra special.

Warm wishes,
The [Brand Name] Team"

Scenario: Feedback Request (Survey Follow-Up)

Subject: "We Value Your Feedback, [First Name]"

Main Content:
"Hi [First Name],
Thank you for being a valued customer! Your opinion matters to us.

Could you take 2 minutes to share your thoughts? Your feedback helps us improve and serve you better.

Click [here] to complete the survey.

As a thank-you, enjoy [specific incentive, e.g., a discount or gift] for your next purchase!

We appreciate your time and input.

Best regards,
The [Brand Name] Team"
'''

if 'initialized' not in st.session_state:
    st.session_state.messages = []  # Reset messages
    st.session_state.messages.append({"role": "system", "content": SYSTEM_PROMPT})
    st.session_state.initialized = True

# Page configuration
st.set_page_config(page_title="ENGAGE", page_icon="🤝", layout="wide")

# API key input
api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
if not api_key:
    st.error("⚠️ Please enter your OpenAI API key to use ELEVATE 📈")
    st.stop()
else:
    helper = Helper(api_key=api_key)

# Main interface
st.title("🤝 ENGAGE: Creating highly engaging, personalized campaigns")
st.sidebar.markdown("<small>*by [cherry](https://www.linkedin.com/in/cherrymaegrace)*</small>", unsafe_allow_html=True)

# Add caching for dataset loading
@st.cache_data
def load_datasets():
    customers = pd.read_csv('https://raw.githubusercontent.com/cherrymaegrace/AIR_Projects/refs/heads/main/05_engage_nlg_rag/datasets/customer_data.csv')
    products = pd.read_csv('https://raw.githubusercontent.com/cherrymaegrace/AIR_Projects/refs/heads/main/05_engage_nlg_rag/datasets/product_data.csv')
    return customers, products

# Replace direct dataset loading with cached function
customers, products = load_datasets()

# Add caching for NLG processing
@st.cache_data
def process_nlg(df, document_column):
    return helper.nlg(df, document_column)

customers_nlg = process_nlg(customers, 'document')
products_nlg = process_nlg(products, 'document')

# Add at the start of the file, after initializing session state
if 'customers_index' not in st.session_state:
    st.session_state.customers_index = None
if 'products_index' not in st.session_state:
    st.session_state.products_index = None

# After loading datasets, precompute embeddings
@st.cache_data
def precompute_embeddings(documents):
    return helper.generate_embeddings(documents)

# Precompute embeddings for both datasets
customers_documents = customers_nlg['document'].to_list()
products_documents = products_nlg['document'].to_list()

if st.session_state.customers_index is None:
    st.session_state.customers_index = precompute_embeddings(customers_documents)
if st.session_state.products_index is None:
    st.session_state.products_index = precompute_embeddings(products_documents)

# Create columns for filters
col1, col2, col3 = st.columns(3)

with col1:
    segment = st.selectbox(
        'Customer Segment',
        options=['High Spender', 'Budget Spender', 'Others']
    )
    
    if segment == 'Others':
        custom_segment = st.text_input('Enter Segment')
        segment = custom_segment if custom_segment else segment

with col2:
    category = st.selectbox(
        'Purchase Category', 
        options=sorted(products['purchase_category'].unique().tolist()) + ['Others']
    )
    
    if category == 'Others':
        custom_category = st.text_input('Enter Purchase Category')
        category = custom_category if custom_category else category

with col3:
    num_results = st.selectbox(
        'Number of Search Results',
        options=[1, 2, 3, 4, 5],
        index=0
    )

email_content = st.text_input('Email Content')

# Batch process emails instead of one at a time
if st.button('Generate Personalized Emails'):
    with st.spinner('Generating personalized emails...'):
        customer_segment = f"{segment} customers who recently purchased {category}"
        customers_search_results = helper.generate_context(
            customer_segment, 
            st.session_state.customers_index, 
            customers_documents, 
            k=max(num_results * 2, 10)  # Increase search space to ensure enough results after filtering
        )
        
        # Ensure we have exactly num_results
        customers_search_results = customers_search_results[:num_results]

        results = []
        product_segment = f"Products in the {category} category for {segment} customers"
        # Get product recommendations once for all customers
        products_search_results = helper.generate_context(
            product_segment, 
            st.session_state.products_index, 
            products_documents, 
            2
        )

        # Prepare all messages in batch
        batch_messages = []
        for customer in customers_search_results[:num_results]:
            context = str(customer) + ' '.join(products_search_results)
            user_message = f'''
            Email Content: {email_content}\n
            Customer: {customer}\n
            Products: {products_search_results}\n
            '''
            structured_prompt = f'Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:'
            batch_messages.append(structured_prompt)

        # Process all messages in one call
        batch_responses = helper.call_model(
            st.session_state.messages, 
            ['user'] * len(batch_messages), 
            batch_messages
        )

        # Process responses
        for i, (customer, response) in enumerate(zip(customers_search_results, batch_responses)):
            user_id = customer.split("\"User ")[1].split(",")[0]
            st.markdown(f"**Generated Email for {user_id}:**")
            st.markdown(response)
            st.markdown("---")