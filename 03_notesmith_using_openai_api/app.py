import streamlit as st
import pandas as pd
import openai
import pdfplumber
from streamlit_option_menu import option_menu
from pdfplumber.utils import extract_text, get_bbox_overlap, obj_to_bbox

#region HELPER FUNCTIONS
def get_text_from_pdf(pdf_file):
    pdf = pdfplumber.open(pdf_file)
    all_text = []

    for page in pdf.pages:
        filtered_page = page
        chars = filtered_page.chars

        for table in page.find_tables():
            first_table_char = page.crop(table.bbox).chars[0]
            filtered_page = filtered_page.filter(lambda obj: 
                get_bbox_overlap(obj_to_bbox(obj), table.bbox) is None
            )
            chars = filtered_page.chars

            df = pd.DataFrame(table.extract())
            df.columns = df.iloc[0]
            markdown = df.drop(0).to_markdown(index=False)

            chars.append(first_table_char | {"text": markdown})

        page_text = extract_text(chars, layout=True)
        all_text.append(page_text)

    pdf.close()
    return "\n".join(all_text)

def call_model(messages):
    system_prompt = """
        You are NoteSmith, an intelligent study assistant designed to help students transform their uploaded notes into tailored Q&As and concise summaries. Your primary task is to analyze the provided notes, generating relevant questions, answers, and summaries based on that material. You can adjust the difficulty of questions to suit the student’s needs, ranging from basic comprehension to more advanced critical thinking.
        While you prioritize content from the uploaded notes, you can provide additional information beyond the notes if explicitly asked. When a student asks for examples, real-world applications, or extra context not covered in the material, feel free to expand thoughtfully while staying relevant to the topic. For instance, if a student asks, "Can you give real-world examples for this?" you may provide examples or explanations that enhance understanding.
        Maintain a professional yet supportive and friendly tone, and aim to enhance clarity, simplify complex ideas, and support the student in mastering their study material.
    """
    
    system_msg = {
        "role": "system",
        "content": system_prompt
    }
    
    # Ensure we always include the system message and context
    if messages[0]["role"] != "system":
        messages = [system_msg] + messages
        
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content
#endregion

st.set_page_config(page_title="NoteSmith", page_icon="🗡️", layout="wide")

st.sidebar.image("images/note-smith-logo.png", width=280)

# Add API key input in the sidebar
api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
if api_key and len(api_key) != 164:
    st.sidebar.error("Invalid API key. It should be 164 characters long.")

# Create an option menu in the sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="",
        options=["NoteSmith","Crafting Guide"]
    )

    st.markdown("""
        *Cherry Mae Caracas ⚔️ AI Republic Student
        Connect with me on [LinkedIn](https://www.linkedin.com/in/cherrymaegrace)*
    """)

# Display the selected scenario
if selected == "Crafting Guide":
    st.header("Crafting Guide: How to begin forging knowledge with NoteSmith?")
    st.subheader("Scenario 1: Reviewing for an Exam")
    st.markdown("""
        **Student:** "I need help reviewing my chemistry notes on acids and bases. Can you make some Q&As?"
        
        **NoteSmith:** "Certainly! I'm analyzing your notes... Here are a few crafted questions to sharpen your understanding:
        
        - What is the difference between a strong acid and a weak acid?
        - How does the pH scale measure acidity? 
        - What is the role of conjugate acids and bases in equilibrium reactions?"
        
        **Student:** "These are great. Can I have more detailed ones?"
        
        **NoteSmith:** "Absolutely. Here are some advanced questions:
        
        - Explain how Le Chatelier's Principle applies to acid-base equilibria.
        - How would adding a common ion affect the dissociation of a weak acid in solution?
        - What's the impact of buffer solutions on pH changes during titration?"
    """)

    st.subheader("Scenario 2: Preparing a Group Study Guide")
    st.markdown("""
        **Student:** "I've uploaded a few PDFs from our group project. Can you create a Q&A set from them?"
        
        **NoteSmith:** "Got it! I'm extracting the key points from all the documents… Here's a collaborative study set:
        
        - What are the main findings from the research on climate change's economic impact?
        - How do mitigation strategies differ between developed and developing countries?
        - What role does renewable energy play in reducing global carbon emissions?"
        
        **Student:** "Looks good. Can we add some more questions that challenge critical thinking?"
        
        **NoteSmith:** "Sure! Here are some deeper questions to provoke thought:
        
        - If current policies remain unchanged, what long-term economic effects could we expect from climate change?
        - How might the transition to renewable energy affect global trade dynamics?
        - What ethical considerations arise in balancing economic growth and environmental sustainability?"
    """)

    st.subheader("Scenario 3: Quick Concept Clarification")
    st.markdown("""
        **Student:** "I'm struggling with the section on quantum mechanics. Can you break it down and create a few basic questions?"
        
        **NoteSmith:** "Of course! I'll simplify the key concepts. Here are a few questions:
        
        - What is the Heisenberg Uncertainty Principle?
        - How does the concept of superposition differ from classical physics?
        - What is meant by wave-particle duality?"
        
        **Student:** "That helps! Can I get a quick summary of superposition too?"
        
        **NoteSmith:** "Certainly! Superposition is the principle that a quantum system can exist in multiple states at once until it is observed or measured, at which point it collapses into a single state."
    """)

if selected == "NoteSmith":
    st.title("⚒️ NoteSmith: Where notes become knowledge")
    st.write("""
         
        NoteSmith is a digital assistant designed to help students transform their raw study materials into polished, refined sets of Q&As, much like a goldsmith shapes precious metals into beautiful pieces.
        With precision and care, NoteSmith forges insightful questions and answers from any academic PDF, turning dense information into interactive, organized learning aids.
        Whether you're working with textbooks, lecture notes, or research papers, NoteSmith takes the rough materials of your studies and shapes them into something valuable for your academic success.

    """)
    st.divider()
    st.header("Upload your notes, and together we'll forge the path to exam success!")

    # Initialize chat history and context in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "context_loaded" not in st.session_state:
        st.session_state.context_loaded = False
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # File uploader
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_file is not None:
        if st.button("Initiate forging sequence"):
            st.write("Processing PDF...")
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                try:
                    openai.api_key = api_key
                    pdf_file.seek(0)
                    
                    # Extract text and initialize context
                    extracted_text = get_text_from_pdf(pdf_file)
                    
                    # Initialize chat with context
                    if not st.session_state.context_loaded:
                        # Store the context separately
                        st.session_state.pdf_context = extracted_text
                        
                        # Create initial message with context
                        context_msg = {
                            "role": "user", 
                            "content": f"Here's the PDF content to analyze: {extracted_text}\n\nPlease acknowledge that you've received this content."
                        }
                        
                        st.session_state.messages = [context_msg]
                        
                        # Get initial response from the model
                        response = call_model(st.session_state.messages)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                        st.session_state.context_loaded = True
                        st.rerun()

                except Exception as e:
                    st.error(f"Error in extraction process: {str(e)}")

    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if st.session_state.context_loaded:
        user_input = st.chat_input("Have questions? Let's shape them into clarity.")
        if user_input:
            # Construct the message with context reminder
            enhanced_input = f"""Based on the PDF content provided earlier, please answer this question: {user_input}"""
            
            # Add user message to both message lists
            st.session_state.messages.append({"role": "user", "content": enhanced_input})
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            # Show loading message
            with st.chat_message("assistant"):
                with st.spinner("Give me a moment to craft your results..."):
                    # Get model response
                    response = call_model(st.session_state.messages)
                    
                    # Add assistant response to both message lists
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
            
            st.rerun()