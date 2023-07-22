import gradio as gr
import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_KEY')

# Load all text documents and split text into smaller chunks
loader = DirectoryLoader("./work-policies", glob="*.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

# Create embeddings to encode and decode inputs and place in vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Track chatbot conversation for conversational retrieval
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# Start retrieval chain for questions 
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever(), memory=memory)

with gr.Blocks(title="OpenAI Turbo GPT-3.5 (ChatGPT)") as demo:
    # ChatBot UI
    chatbot = gr.Chatbot().style(height=500)
    user_input = gr.Textbox(label="Message input", placeholder="Ask your question to send to ChatGPT. Press enter to submit your message.")
    clear = gr.Button("Clear Chat")

    examples = gr.Examples(examples=[["Who are you?"], ["What are my workplace policies?"],["When can I go on holiday?"], ["What are my rights in the workplace?"], ["Explain my health insurance."], ["Explain my working hours and attendance."], ["Tell me a fun story."], ["What is 7+7?"]], inputs=[user_input])
    # Track all user and assistant messages for subsequent answers
    messages = []
      
    def bot(history):
        bot_response = qa({"question": messages[len(messages)-1]})
        print(bot_response)
        # Stream message to chatbot by queueing characters until a full response is yielded
        history[-1][1] = ""
        for char in bot_response["answer"]:
            history[-1][1] += char
            yield history
        
    def user(input_text, history):
        messages.append(input_text)
        # Return response to chatbot and clear textbox and set its interaction to false while awaiting response
        return gr.update(value="", interactive=False), history + [[input_text, None]]    

    # Submit user input to chatbot and set textbox to be interactive
    user_input.submit(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
            bot, [chatbot], [chatbot]).then(lambda: gr.update(interactive=True), None, [user_input], queue=False)
    
    # Clear chat history and user input
    clear.click(lambda: None, None, [chatbot], queue=False).then(lambda: None, None, [user_input], queue=False)

demo.queue()
demo.launch(server_port=8080)