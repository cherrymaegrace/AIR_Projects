# ENGAGE - AI-Powered Email Marketing Personalization

## ✨ Try it out!

Experience ENGAGE in action! Click the link below to access our live demo:

🔗 [Live Demo](https://engage.streamlit.app)

## Overview
ENGAGE is a sophisticated email marketing tool that leverages RAG (Retrieval Augmented Generation) and AI to create highly personalized email campaigns. The application combines customer segmentation data with purchase behavior to generate targeted, engaging email content for various marketing scenarios.

## Features
- 🎯 Customer Segmentation: Filter and target specific customer segments
- 📊 Data-Driven Insights: Utilize customer behavior and purchase history
- ✍️ AI-Powered Content Generation: Create personalized email content at scale
- 🔍 Smart Retrieval: Find relevant customer contexts and product recommendations
- 💌 Multiple Email Scenarios: Support for onboarding, reactivation, upselling, seasonal promotions, and feedback requests

## Prerequisites
- Python 3.8+
- OpenAI API key
- Required Python packages:
  ```bash
  pip install streamlit langchain openai pandas numpy chromadb
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd engage-email-marketing
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Enter your OpenAI API key in the sidebar
2. Select customer segment and purchase category filters
3. Choose the number of email results to generate
4. Input your desired email content
5. Click "Generate Personalized Emails" to create targeted content

## Data Requirements
The application expects two CSV datasets:
- `customer_data.csv`: Customer segmentation and behavior data
- `product_data.csv`: Product catalog and category information

## Technical Architecture
- Frontend: Streamlit
- NLP/AI: LangChain + OpenAI
- Vector Store: Chroma
- Data Processing: Pandas, NumPy
- Caching: Streamlit's built-in caching mechanism

## Features in Detail
### Customer Segmentation
- High Spender
- Budget Spender
- Custom segments

### Email Contents
- Welcome emails
- Reactivation campaigns
- Upsell promotions
- Seasonal campaigns
- Feedback requests

## Performance Optimization
- Cached dataset loading
- Precomputed embeddings
- Batch processing for multiple emails
- Smart context retrieval

## Author
Created by [cherry](https://www.linkedin.com/in/cherrymaegrace)

## Acknowledgments
- OpenAI for the language model
- LangChain for the RAG implementation
- Streamlit for the web interface
