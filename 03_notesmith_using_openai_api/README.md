# NoteSmith App

NoteSmith is an intelligent study assistant built on Streamlit that leverages OpenAI's powerful GPT models through their API. This chatbot analyzes uploaded PDF files, creates customized Q&As, and summarizes topics based on the content of the PDF. Using OpenAI's advanced natural language processing capabilities, it can understand context, generate relevant questions, and provide detailed explanations tailored to the study material.

## Features

- **PDF Analysis:** Upload PDF files for NoteSmith to extract text and analyze content.
- **Q&A Generation:** Automatically generate relevant questions and answers based on the material in the uploaded notes.
- **Summarization:** Create concise summaries of key topics covered in the PDF files.
- **User-Friendly Interface:** Interact with NoteSmith through an intuitive chat interface.

## How to Use

1. **Upload Your PDF:** Start by uploading your PDF file containing notes or study material.
2. **Chat with NoteSmith:** Ask questions or request summaries, and NoteSmith will respond with tailored insights.
3. **Receive Q&As:** NoteSmith will generate Q&A sets from the uploaded content for effective studying.

## Technicalities

- **PDF Text Extraction:** Uses pdfplumber library to efficiently extract text content from uploaded PDF files before processing with OpenAI.
- **Prompting Framework:** The system implements the RICCE prompting framework for effective AI interactions:
  - **R**ole: Defines the AI assistant's persona and expertise
  - **I**nstructions: Clear directives for the desired output
  - **C**ontent: Input material to be processed
  - **C**onstraints: Boundaries and limitations for the response
  - **E**xamples: Sample inputs and outputs for reference
