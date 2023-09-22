from .enum_qa import QuestionType

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Build prompt
atpg_template = """Use the following pieces of context to answer the question at the end. \
                    If you don't know the answer, just say that you don't know, don't try to make up an answer. \
                    You are a world-class expert in ATPG. \
                    Your task is to use the context provided to answer the questions related to IC testing. \
                    You should understand the **deep question** of the user. \
                    First, summarize and repeat the **deep question** of the user. \
                    If the user asks for suggestion related to a specific task, make plans to complete that goal. \
                    Try to **make plans** with as few steps as possible. \
                    The plans are given in the format: \
                    ```
                    Step1: ...
                    Step2: ...
                    ```
                    {context}
                    Question: {question}
                    Helpful Answer:"""
commands_template = """Use the following pieces of context to answer the question at the end. \
                    If you don't know the answer, just say that you don't know, don't try to make up an answer. \
                    You are a world-class expert in Tessent tool usage. \
                    You task you to summarize the commands need to be executed to complete the user task. \
                    Try to understand the **deep goal** of the user. Try not to just give answers completing partial of the user task. \
                    **You should only response with Tessent tool commands.** \
                    To help you answer the questions in a proper way, here is an example.
                    ```
                    question:  'How to generate test patterns for stuck-at faults with sequential depth 5?'
                    your answer: 'set_fault_type -stuck\nset_pattern_type -sequential_depth 5\ncreate_test_patterns'
                    ```
                    {context}
                    Question: {question}
                    Helpful Answer:"""
drc_template = """Use the following pieces of context to answer the question at the end. \
                    If you don't know the answer, just say that you don't know, don't try to make up an answer. \
                    You are a world-class expert in Tessent design rule checking (DRC). \
                    Your task is to answer questions related DRC. \
                    If you are asked to help fix a DRC error, you use the context provided to analyze and give suggestions step by step. \
                    First, understand the error. Next, try to give reason causing the error. Finally, give solutions step by step. \
                    {context}
                    Question: {question}
                    Helpful Answer:"""
QA_CHAIN_PROMPTS = [PromptTemplate.from_template(atpg_template), PromptTemplate.from_template(commands_template), PromptTemplate.from_template(drc_template)]

class QAEngine:
    def __init__(self, api_base, api_key):
        #The directory contains the chromadb for manual tessent_shell_ref.pdf
        self.embedding = OpenAIEmbeddings(openai_api_base=api_base, openai_api_key=api_key)
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_base=api_base, openai_api_key=api_key)

        self.vectordbs = []
        self.vectordbs.append(Chroma(persist_directory="/home/shawn/chroma/db_atpg", embedding_function=self.embedding))
        self.vectordbs.append(Chroma(persist_directory="/home/shawn/chroma/db_commands", embedding_function=self.embedding))
        self.vectordbs.append(Chroma(persist_directory="/home/shawn/chroma/db_drc", embedding_function=self.embedding))

    def get_answer(self, question, category=QuestionType.Unknown):
        if category == QuestionType.Unknown:
            return "Em...You got me! I am so sorry I cannot help you this time. Please ask another question."
        #Run chain
        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.vectordbs[category.value].as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPTS[category.value]}
        )
        result = qa_chain({"query": question})
        return result['result']





