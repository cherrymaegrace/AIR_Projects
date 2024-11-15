import os
import openai
import numpy as np
import pandas as pd
import json
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
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

#region Prompts
SYSTEM_PROMPT = """
Role:
You are BlockMate, an intelligent assistant designed to manage foam block inventory and logistics. Your primary role is to provide real-time stock updates, track inventory levels, recommend optimal movement sequences for foam blocks, and assist with planning by giving proactive insights based on foam grades, stock levels, and usage trends. You will reference and accurately interpret the foam block dataset provided.

Instructions:

Respond to user queries concisely and accurately by referencing the dataset. You will provide information regarding foam block inventory, stock movement, and logistics.
When a user asks how many blocks of a specific foam grade are in stock, reference the batch_number, foam_grade, color, and size measurements (which include width, height, and length) for each block in stock. Always ensure that these details match exactly with the dataset.
Provide real-time data on current foam block stock, alert users to low or high stock levels, and offer foam grade-specific details.
Suggest efficient block movement sequences based on the block's location in the warehouse (from the location column), the block's size, and production requirements.
Proactive alerts such as reorder recommendations or overstock warnings should be based on stock levels, historical trends, and upcoming production forecasts.
Adjust responses based on user role (e.g., planners need inventory forecasts, operators need movement-specific instructions).
Ensure all responses use the actual values from the dataset (e.g., correct foam grades, sizes, and locations) and display the details in an easy-to-read format.
Use professional yet approachable language, avoiding unnecessary jargon unless the user is familiar with technical terms.
Context:
BlockMate operates in a fast-paced manufacturing environment where foam blocks of various grades and sizes are regularly moved and utilized. Accurate tracking of inventory is crucial to avoid overstocking or understocking, which can lead to inefficiencies or production delays. Users rely on you to deliver the right data at the right time, enabling optimized workflows and planning.

Key user groups include:

Planning Teams: Focus on overall inventory levels, stock forecasting, and avoiding production schedule disruptions.
Warehouse Operators: Focus on physical movement of foam blocks and optimizing storage space.
Production Managers: Ensure the right foam grades are available for upcoming production runs.
Dataset Information:
You will reference a foam block inventory dataset with the following columns:

batch_number – A unique identifier for each foam block in the format ddmmyyyyLetter (e.g., 101023A), where the date is in ddmmyyyy format and the letter is A, G, or L.
foam_grade – Categorical names representing different foam types, themed around "Good Sleep and Reduce Back Pain" (e.g., "Dream Support", "Orthopedic Ease", "SpineAlign"). There are 10 possible foam grades.
color – Categorical values indicating foam color (e.g., "White", "Blue", "Pink", "Gray", "Yellow").
width – Categorical values for foam width (e.g., 75, 77, 80, 83, 85 inches).
height – Random values between 20 and 25 inches.
length – Always 60 meters.
receive_date – The date the foam block was received, formatted as dd-mm-yyyy. It must be within 10 days of the batch_number date.
location – A unique identifier for the foam block's storage position in the warehouse, formatted as LaneStack (e.g., A1, B3), where Lane is a letter from A to W and Stack is a number from 1 to 4. No duplicate locations are allowed.
operator – The name of the operator who handled the foam block.
max_temp – A random value between 70 and 90 representing the maximum temperature the foam block can withstand during storage.
Constraints:

Ensure data accuracy – Always provide real-time and accurate data from the dataset. Avoid vague or outdated information.
When listing foam blocks by foam grade, ensure there are no duplicate locations in the dataset. If there is a duplicate location, return an error.
Keep responses concise – Avoid overwhelming users with unnecessary details; focus on the most relevant data.
Avoid speculative answers – Base all responses on available data or forecast trends.
Movement Recommendations – When suggesting block movements, prioritize logistical efficiency, avoid congestion, and minimize unnecessary handling.
Prioritize production needs – When recommending block movements, consider foam grade availability and available storage space for upcoming production runs.
Examples:

Example 1 (Inventory Query):

User: "BlockMate, how many blocks of grade 'Dream Support' foam do we have in stock?"
BlockMate:
"You currently have 4 blocks of 'Dream Support' foam in stock. Here are the details for each block:
Batch #101023A, Color: White, Size: 75x22x60, Location: A2
Batch #121023G, Color: Blue, Size: 80x21x60, Location: B1
Batch #151023A, Color: Pink, Size: 83x25x60, Location: C3
Batch #181023L, Color: Gray, Size: 85x23x60, Location: D4"
Example 2 (Movement Recommendation):

User: "What's the best sequence to move the Grade 'Orthopedic Ease' blocks for the next batch?"
BlockMate: "To optimize the workflow, start by moving the Grade 'Orthopedic Ease' blocks from Location A1, followed by B2. This minimizes handling time and avoids congestion for incoming stock."
Example 3 (Alert):

User: "Any updates on stock levels?"
BlockMate: "You are nearing critical stock for Grade 'SpineAlign' foam, with only 20 blocks remaining. Based on production needs, I recommend placing a reorder within the next 3 days."
"""
#endregion

#region Helpers

# Concatenate all column values for a row
def concat_all_columns(row):
    return ' '.join(row.values.astype(str))

# Call model
def call_model(role: list = [], prompt: list = []):
    additional_message = []

    if role and prompt and len(role) == len(prompt):
        for i in range(len(role)):
            additional_message.append({'role': role[i], 'content': prompt[i]})

    chat = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=st.session_state.messages + additional_message,
        temperature=0.5,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    
    return chat.choices[0].message.content

# Generate embeddings and vector store index
def generate_dataset_embeddings(dataset):
    # Combine all column values of the row to a single column
    dataset['combined'] = dataset.apply(concat_all_columns, axis=1)
    # Create documents list from the combined values
    documents = dataset['combined'].tolist()
    # Generate embeddings for the combined values
    embeddings = [get_embedding(doc, engine='text-embedding-3-small') for doc in documents]
    # Convert emebeddings into an array
    embeddings_np = np.array(embeddings).astype('float32')
    # Initialize vector store index
    index = faiss.IndexFlatL2(len(embeddings[0]))
    # Add embeddings array into the vector store
    index.add(embeddings_np)

    return index, documents

# Generate context
def generate_context(user_message, index, documents):
    query_embeddings = get_embedding(user_message, engine = "text-embedding-3-small")
    query_embeddings_np = np.array([query_embeddings]).astype('float32')
    _, indices = index.search(query_embeddings_np, 10)
    retrieved_documents = [documents[i] for i in indices[0]]
    context = ' '.join(retrieved_documents)
    structured_prompt = f'Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:'

    return structured_prompt

#endregion

# Page configuration
st.set_page_config(page_title="BlockMate: Foam Inventory Assistant", page_icon="🛏️", layout="wide")

# Sidebar setup
st.sidebar.image("images/blockmate-logo.png")

# API key input
api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
if not api_key:
    st.error("⚠️ Please enter your OpenAI API key to use BlockMate")
    st.stop()
else:
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

# Main interface
st.title("🛏️ :blue[Welcome to BlockMate!]")
st.markdown("""
Say goodbye to inventory confusion and stock management headaches. BlockMate is your intelligent foam block inventory assistant, designed to streamline your workflow and keep your warehouse running smoothly.

With BlockMate, you can:

- Track Inventory in Real-Time: Get up-to-the-minute updates on foam block levels, locations, and stock status.
- Visualize Your Stock Effortlessly: See your inventory mapped on a 2D grid for clear, easy tracking.
- Optimize Block Movement: Receive smart recommendations on how to efficiently move and organize foam blocks, reducing unnecessary handling.
- Avoid Overstocking and Understocking: Stay ahead of demand with proactive alerts, ensuring you always have the right foam grades on hand.

**Your operations just got smarter!**

BlockMate is here to make sure you never miss a beat when it comes to managing your foam block inventory, keeping your production flowing seamlessly. Ready to optimize your warehouse? Let's get started!
""")

# Add file uploader
uploaded_file = st.sidebar.file_uploader("Upload Foam Inventory CSV", type=['csv'])
if not uploaded_file:
    st.error("⚠️ Please upload your foam inventory CSV file")
    st.stop()

# Load the CSV file
try:
    stocks = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")
except Exception as e:
    st.error(f"Error loading CSV file: {str(e)}")
    st.stop()

st.sidebar.markdown("<small>*by [cherry](https://www.linkedin.com/in/cherrymaegrace)*</small>", unsafe_allow_html=True)

#region Create a visual representation of the inventory
lanes = list('ABCDEFGHIJKLMNOPQRSTUVWX')  # 24 lanes (A to W)
stacks = [1, 2, 3, 4]  # 4 stacks

# Initialize the grid as empty
grid = np.full((len(stacks), len(lanes)), '', dtype=object)

# Place each block on the grid and ensure unique locations
for index, row in stocks.iterrows():
    location = row['location']
    lane = location[0]  # The letter (A-W) corresponds to the lane
    stack = int(location[1])  # The number (1-4) corresponds to the stack

    # Find the index positions for the grid
    x = lanes.index(lane)  # X-axis (LANE)
    y = len(stacks) - stack  # Y-axis (STACK), reversed so stack 1 is closest to x-axis

    # Check for duplicates
    if grid[y, x] != '':
        st.warning(f"Duplicate location detected at {location}.")
        continue
    
    # Fill the grid with the relevant data
    grid[y, x] = f"{row['batch_number']}\n{row['foam_grade']}\n{row['color']}\n{row['width']}x{row['height']}x{row['length']}"

# Plot the grid
fig, ax = plt.subplots(figsize=(24, 4))  # Further reduced height to 4

# Add the labels for the blocks
for y_val in range(grid.shape[0]):
    for x_val in range(grid.shape[1]):
        if grid[y_val, x_val]:  # Only display non-empty entries
            ax.text(x_val, y_val, grid[y_val, x_val], 
                   va='center', 
                   ha='center', 
                   fontsize=9,     # Increased font size
                   bbox=dict(facecolor='white', 
                           alpha=0.9,
                           pad=0.3,    # Further reduced padding
                           boxstyle='round,pad=0.2'))  # Reduced box padding

# Set axis ticks and labels
ax.set_xticks(np.arange(len(lanes)))
ax.set_yticks(np.arange(len(stacks)))
ax.set_xticklabels(lanes, fontsize=12)
ax.set_yticklabels(reversed(stacks), fontsize=12)

# Adjust the plot limits to reduce spacing
ax.set_xlim(-0.5, len(lanes) - 0.5)
ax.set_ylim(len(stacks) - 0.5, -0.5)

# Add more space between subplots to prevent text overlap
plt.tight_layout(pad=0.5)  # Further reduced padding in tight_layout

# Adjust grid appearance
ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
plt.title('Foam Block Inventory Representation (LANE x STACK)', fontsize=14, pad=20)

st.pyplot(fig)

#endregion

if 'initialized' not in st.session_state:
    st.session_state.messages = []  # Reset messages
    st.session_state.messages.append({"role": "system", "content": SYSTEM_PROMPT})
    st.session_state.initialized = True

index, documents = generate_dataset_embeddings(stocks)

# Display existing messages
for message in st.session_state.messages:
    if message['role'] != 'system':  # Skip system messages
        with st.chat_message(message['role']):
            st.markdown(message['content'])

# Handle new user input
if user_message := st.chat_input('Ask BlockMate!'):
    # Process user message
    def add_message(role, content):
        st.session_state.messages.append({'role': role, 'content': content})
        with st.chat_message(role):
            st.markdown(content)
    
    # Add and display user message
    add_message('user', user_message)
    
    # Generate context and create a complete prompt
    structured_prompt = generate_context(user_message, index, documents)
    
    # Make the API call with the structured prompt
    response = call_model(
        role=['user'],
        prompt=[structured_prompt]
    )
    add_message('assistant', response)

