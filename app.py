import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_groq import ChatGroq
import os
import random

load_dotenv()
DB_PATH = "chromadb"
embedding_model_name="nomic-embed-text"
embeddings = OllamaEmbeddings(model=embedding_model_name)

# initializing groq api from the .env file
groq_api_key = os.environ['GROQ_API_KEY']

#list of the GROQ API's model names
models = {
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it"
}

set_of_questions = {
     'What is East West University?',
     'Where is East West University located?',
     'Who is the founder of East West University?',
     'Who is the Vice Chancellor of East West University?',
     'How Many Faculties are there in East West University?',
     'What is the Semester System of EWU',
     'Provide Class System of EWU',
     'What are the Admission Eligibilites except B.Pharm?',
     'What are the Admission Eligibilites for B.Pharm',
     'What are the degrees offered for the Undergraduates?',
     'What are the degrees offered for the Graduates?'
}

set_of_tables = {
    'table of the tuition fee per credit of all programs of Undergraduates',
    'table of the tuition fee per credit of all programs of Graduates',
    'table of the tuition fee details of all programs of Undergraduates',
    'table of the tuition fee details of all programs of Graduates',
    'table of Course Flowchart of CSE',
    'table of Course Summary of CSE' 
}

set_of_list = {
    'List of the Members of Board of Trustees',
    'List of the Syndicate Members',
    'List of the Academic Council Members',
    'List of the Faculty Members of CSE',
    'List of the Faculty Members of EEE',
    'List of the Faculty Members of GEB',
    'List of the Faculty Members of Pharmacy',
}

test_questions = [
    "how many faculties are there in ewu?",
    "what is ewu?",
    "what is the mission of ewu?",
    "what is the vision of ewu?",
    "where is ewu located?",
    "what are the degrees do they offer in ewu for undergrads?",
    "what are the degrees do they offer in ewu for grads?",
    "Who is the Vice Chancellor of East West University?",
    "Who is the founder of East West University?",
    "List of the Syndicate Members",
    "what is the class System of EWU",
    "provide semester system of ewu",
    "What are the Admission Eligibilites for B.Pharm",
    "What are the Admission Eligibilites except B.Pharm?",
    "Make a Table of the tuition fee per credit of all programs of Undergraduates",
    "Make a Table of the tuition fee per credit of all programs of Graduates",
    "Make a Table of the tuition fee details of all programs of Graduates",
    "give me the table of tuition fee details for undergrads",
    "give me the course summary of cse",
    "how many majors are there in cse department?",
    "what are the course of data science major?",
    "what are the courses of software major for undergrads?",
    "provide the chairperson names of all departments of science faculty",
    "give me the list of faculty members of cse",
    "is cse366 a core course of cse?",
    "what is the name of the course cse366?",
    "what is the prerequisite for the course cse366?",
    "what is the prerequisite for the course cse405?",
    "what is the name of the course cse303",
]

#function for showing random quesitons after each query
def showRandomQuestions():
    random_ques_seq1 = random.randint(0, 9)
    random_ques_seq2 = random.randint(10, 19)
    random_ques_seq3 = random.randint(20, len(test_questions))
    
    random_ques1 = test_questions[random_ques_seq1]
    random_ques2 = test_questions[random_ques_seq2]
    random_ques3 = test_questions[random_ques_seq3]


    with st.container(border=True):
        st.markdown('###### Next questions you may ask:')
        col1, col2, col3 = st.columns(3)
        with col1:
                st.write(random_ques1)
        with col2:
                st.write(random_ques2)
        with col3:
                st.write(random_ques3)

#function for initializing GROQ API
def initializeGROQ(api_key, model_name, chatbot_interaction_level):
     llm_groq = ChatGroq(
                    groq_api_key=api_key, 
                    model_name=model_name,
                    temperature=chatbot_interaction_level
                )
     return llm_groq

def main():
     

    st.set_page_config(page_title="EWU RAG Chatbot", page_icon=":book:")
    st.header("EWU RAG Chatbot (Anything related to EWU) :books:")

    #initializing session state variables
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    if "messages" not in st.session_state:
            st.session_state.messages = []

    if "chatbot_level" not in st.session_state:
        st.session_state.chatbot_level = None

    if "vectorDB" not in st.session_state:
        st.session_state.vectorDB = None

    if "chain" not in st.session_state:
        st.session_state.chain = None 

    if "qa" not in st.session_state:
        st.session_state.qa = None

    st.session_state.vectorDB = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    with st.sidebar:
        

        # Layout for model selection and max_tokens slider
        model_option = st.selectbox(
            "Choose a model:",
            options=list(models),
            #format_func=lambda x: models[x]["name"],
            index=1  # Default to mixtral
        )

        # Layout for chatbot interaction level slider
        chatbot_interaction_level = st.slider(
            "Chatbot Interaction Level:",
            min_value=0.1,  # Minimum value to allow some flexibility
            max_value=1.0,
            # Default value or max allowed if less
            value=0.3,
            step=0.1,
            help=f"Set the chatbot's interaction level to your liking. Precise: (0.2-0.3), Moderate:(0.5-0.7), Creative: (0.8 - 1.0)"
        )

        with st.container(border=True):
            st.markdown('## Frequently Asked Questions')
            selectQuestion = st.selectbox(
                'Select a question',
                options=set_of_questions
            )
            askButton = st.button('Ask the selected question')

        
        with st.expander('Generate tables'):
            st.markdown('## Generate Table')
            selectTable = st.selectbox(
                'Select a table to generate',
                options=set_of_tables
            )
            generateTableButton = st.button('Generate selected table')
            
        with st.expander('Generate Lists'):
            st.markdown('## Generate List')
            selectList = st.selectbox(
                'Select a table to generate',
                options=set_of_list
            )
            generateListButton = st.button('Generate selected list')

        # testRAGButton = st.button('Begin test')
    

    # Initializing GROQ chat with provided API key, model name, and settings
    llm_groq = initializeGROQ(groq_api_key, model_option, chatbot_interaction_level)

    # Create a retriever from the Chroma vector database
    retriever = st.session_state.vectorDB.as_retriever(search_kwargs={"k": 5})


    # Create a RetrievalQA from the model and retriever
    st.session_state.qa = RetrievalQA.from_chain_type(llm=llm_groq, chain_type="stuff", retriever=retriever)

    #user query in the variable
    user_question = st.chat_input("Ask about anything related to East West University")


    if askButton:
        user_question = selectQuestion
    if generateTableButton:
        user_question = selectTable
    if generateListButton:
        user_question = selectList


    #printing the session messages
    for message in st.session_state.messages:
            # avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
            with st.chat_message(message["role"]):
                # st.markdown("###### " + message["content"])
                st.write(message["content"])
                model_msg = f'<p style="font-family:Courier; font-size: 12px;">Model: {message["model"]}</p>'
                st.markdown(model_msg, unsafe_allow_html=True)

    #styling part
    model_name_html = f'<p style="font-family:Courier; font-size: 12px;">Model: {model_option}</p>'

    if user_question:
        #  askQuestion(user_question)
        with st.chat_message("User"):
            st.write(user_question)
            #   st.markdown("###### " + user_question)
            st.markdown(model_name_html, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", 
                                        "content": user_question,
                                        "model": model_option})
        try:
            #getting response based on the query
            response = st.session_state.qa.invoke(user_question)
            st.session_state.messages.append({"role": "assistant", 
                                        "content": response['result'],
                                        "model": model_option})
            
            with st.chat_message("AI"):
                st.write(response['result'])
                st.markdown(model_name_html, unsafe_allow_html=True)
                
                showRandomQuestions()
                
        except:
            with st.chat_message("AI"):
                st.write('Something went wrong!. Check if serve is on or the model is unavaiable')

if __name__ == "__main__":
     main()