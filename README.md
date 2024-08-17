### Project Overview

This code implements an "Advanced Research Assistant" using Python and Streamlit, inspired by Stanford's STORM app. The project integrates large language models (LLMs) with various research tools to automate the research process. It allows users to input a research topic, related keywords, a description, and optional PDF files, which are then processed by a series of agents, each with distinct roles. The final output is a comprehensive research report.

### Key Components

1. **Environment Setup:**
   - Environment variables are loaded using `dotenv` for API keys.
   - The project uses the `groq` library to interact with the Groq Llama 3.1 70B model for generating research outputs.

2. **Google Scholar Search:**
   - A custom tool (`google_scholar_tool`) is created to search academic articles on Google Scholar based on user-provided keywords made using the serper api.

3. **PDF Processing:**
   - If PDF files are uploaded, they are processed into a vector database using the `HuggingFaceEmbeddings` and `FAISS` libraries for efficient search and retrieval during research.

4. **Agent-Based Research Workflow:**
   - **Junior Researcher:** Conducts the initial research using tools like Google Scholar.
   - **Senior Researcher:** Validates and synthesizes the findings to ensure quality.
   - **Research Director:** Oversees the entire process and ensures that the final output meets high academic standards.

5. **Task Definition:**
   - The workflow is defined by tasks assigned to each agent. Tasks include conducting research, validating findings, and synthesizing the final report.

6. **Streamlit Interface:**
   - A simple front-end is built using Streamlit to collect user input and display the research results. The interface allows users to input the main research topic, keywords, and a description. Users can also upload PDF files to enhance the research process.

### Challenges & Improvements

- **Performance:** Currently, the research process takes longer than expected (over three minutes). 
- **Interactivity:** The agent conversations are not yet displayed in the front-end. Adding this feature will significantly enhance the user experience by making the process more interactive and transparent.

- ### Required Libraries and APIs

To run the "Advanced Research Assistant" project, you'll need to install the following Python libraries and configure the necessary APIs. Below is a list of the libraries and API keys required:

### Libraries

1. **Streamlit**
   - For building the web interface.
   - Installation: `pip install streamlit`

2. **dotenv**
   - For loading environment variables from a `.env` file.
   - Installation: `pip install python-dotenv`

3. **requests**
   - For making HTTP requests to external APIs (e.g., Google Scholar API).
   - Installation: `pip install requests`

4. **crewai**
   - For creating and managing agents, tasks, and crews.
   - Installation: `pip install crewai`

5. **langchain-groq**
   - For interacting with the Groq Llama 3.1 70B model.
   - Installation: `pip install langchain-groq`

6. **langchain-community**
   - For using community tools, including `PyPDFLoader`, `HuggingFaceEmbeddings`, and `FAISS`.
   - Installation: `pip install langchain-community`

7. **FAISS**
   - For efficient vector search and retrieval.
   - Installation: `pip install faiss-cpu`

### APIs

1. **Groq API**
   - Required for interacting with the Groq LLM (Llama 3.1 70B).
   - Get your API key from Groq and store it in your `.env` file:
     ```
     GROQ_API_KEY=your_groq_api_key
     ```

2. **Google Scholar API (via Serper.dev)**
   - Used for searching academic articles on Google Scholar.
   - Sign up for a Serper.dev API key and store it in your `.env` file:
     ```
     SERP_API_KEY=your_serper_api_key
     ```

### .env File Example

Create a `.env` file in the root directory of your project with the following content:

```plaintext
GROQ_API_KEY=your_groq_api_key
SERP_API_KEY=your_serper_api_key
```

### PS Note 🤓:
This project is a work in progress. The current implementation faces performance issues with outputs taking longer than three minutes. To improve interactivity, future versions will include displaying agent conversations in the front end and maybe in the future i will try to convert it into a webapp or use gradio at the time of deployment. The project was inspired by Stanford's STORM app, and there's still much to be added to achieve the desired functionality and efficiency.

### Final Output of the assistant:

I did a test run of the application and wanted to explore the topic of role of LLM and conversational agents and this was the output of the assistant:
**Evaluation of Large Language Models: A Comprehensive Review**

The rapid advancement of artificial intelligence has led to the development of large language models, which have revolutionized the field of natural language processing (NLP). This review aims to provide a comprehensive evaluation of these models, highlighting their strengths, weaknesses, and future directions.

**Strengths:**

Large language models have demonstrated several strengths, including improved performance, increased efficiency, and enhanced generalizability. These models have achieved state-of-the-art performance in various NLP tasks, such as language translation, text summarization, and sentiment analysis.

**Weaknesses:**

Despite their strengths, large language models also exhibit several weaknesses, including lack of transparency, vulnerability to bias, and high computational requirements. These models can perpetuate and amplify biases present in the training data, leading to unfair outcomes.

**Future Directions:**

To address the weaknesses of large language models and further improve their performance, several future directions are proposed. These include developing techniques to interpret and understand the decisions made by large language models, promoting fairness and equity in NLP applications, and exploring more efficient training methods for large language models.

**Conclusion:**

In conclusion, large language models have revolutionized the field of NLP, demonstrating remarkable performance in various tasks. However, it is essential to acknowledge their weaknesses and address them to ensure their continued improvement and effective deployment. By prioritizing explainability, fairness, and efficiency, we can unlock the full potential of large language models and promote their widespread adoption.

**References:**

Bolukbasi, T., Chang, K. W., Zou, J. Y., Saligrama, V., & Kalai, A. T. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. In Advances in Neural Information Processing Systems (pp. 4349-4357).

Chen, Y., Zhang, Y., & Chen, X. (2016). Efficient training of large-scale neural networks. Journal of Machine Learning Research, 17(1), 1-22.

Collobert, R., Weston, J., Bottou, L., Karlen, M., Kavukcuoglu, K., & Kuksa, P. (2011). Natural language processing (almost) from scratch. Journal of Machine Learning Research, 12(Aug), 2493-2537.

Kim, Y. (2014). Convolutional neural networks for sentence classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (pp. 1746-1751).

Lipton, Z. C. (2018). The mythos of model interpretability. ACM Queue, 16(3), 31-41.

Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., Macherey, W., ... & Klingner, J. (2016). Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144.

Zoph, B., Yuret, D., May, J., & Knight, K. (2016). Transfer learning for low-resource neural machine translation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (pp. 1568-1574).
