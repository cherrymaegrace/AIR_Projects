## Fine-tuning Gemma-2b instruct for Data Science Q&A

This project explores fine-tuning the Gemma-2b instruct model on the soufyane/DATA_SCIENCE_QA dataset from Hugging Face. The goal is to improve the model's ability to answer data science related questions.  

* The Gemma-2b instruct model was fine-tuned on the DATA_SCIENCE_QA dataset using the Hugging Face Transformers library. 
* The fine-tuned model is uploaded and available on the Hugging Face Model Hub. 
* A separate notebook demonstrates how to use the fine-tuned model for inference tasks, allowing users to ask data science questions and receive answers.
* Currently, semantic similarity is used as the evaluation metric for assessing the model's performance. 

**Further Exploration:**

* Explore alternative evaluation metrics more specific to question answering tasks like BLEU score or ROUGE score.
* Investigate different fine-tuning techniques like parameter-efficient fine-tuning for improved resource efficiency.

**Hugging Face Model Hub:** [Finetuned Model](https://huggingface.co/cherrymaecaracas/gemma-2b-it-ft-data-science-qa)