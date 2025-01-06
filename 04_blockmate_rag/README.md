# BlockMate 🛏️

BlockMate is an intelligent foam block inventory assistant powered by AI that helps warehouse managers and operators track, manage, and optimize their foam inventory in real-time.

## ✨ Try it out!

Experience BlockMate in action! Click the link below to access our live demo:

🔗 [Live Demo](https://blockmate.streamlit.app)

Upload a sample inventory CSV file and start exploring the power of AI-driven inventory management!

## 🚀 Features

- **Real-time Inventory Tracking**: Monitor foam block locations, grades, and specifications
- **Visual Warehouse Map**: 2D grid visualization of your warehouse showing block positions
- **Smart Movement Recommendations**: AI-powered suggestions for optimal block movement
- **Contextual Search**: RAG (Retrieval Augmented Generation) enabled search for accurate inventory information
- **Interactive Chat Interface**: Natural language interaction with the inventory system

## 🛠️ Technologies Used

- **OpenAI API**
  - GPT-4 for natural language processing
  - Text Embeddings for semantic search
- **Streamlit**: Web interface
- **LangChain**: RAG implementation
- **FAISS**: Vector similarity search
- **Pandas**: Data manipulation
- **Matplotlib**: Warehouse visualization
- **NumPy**: Numerical operations

## 🚦 Getting Started

1. Clone the repository
2. Install dependencies:
`pip install -r requirements.txt`
3. Set up your OpenAI API key
4. Run the Streamlit app:
`streamlit run app.py`

## 💻 Usage

1. Launch the application
2. Enter your OpenAI API key in the sidebar
3. Upload your foam inventory CSV file
4. Use the chat interface to:
   - Query inventory levels
   - Get block locations
   - Request movement recommendations
   - Check stock status
   - Monitor temperature conditions

## 📝 Example Queries

- "How many blocks of 'Dream Support' foam do we have in stock?"
- "What's the best sequence to move the Grade 'Orthopedic Ease' blocks?"
- "Show me all white foam blocks in Lane A"
- "Are there any blocks exceeding temperature limits?"

## 🔒 Security

- API keys are handled securely through Streamlit's secure input
- Data is processed locally and not stored permanently
- User authentication can be implemented for production use

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Credits

Developed by [cherry](https://www.linkedin.com/in/cherrymaegrace)

## 📧 Support

For support and queries, please open an issue in the repository.
