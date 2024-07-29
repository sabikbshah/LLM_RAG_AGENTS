
#
import pandas as pd
import os
import dotenv
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Reading a CSV file
csv_file_path = 'stocklive2.csv'
df_csv = pd.read_csv(csv_file_path)
print(df_csv)

# Reading an Excel file
#excel_file_path = 'your_file.xlsx'
#df_excel = pd.read_excel(excel_file_path)

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
print(openai_api_key)



groq_api = os.getenv('GROQAPI_KEY')


print(groq_api)
os.environ['GROQAPI_KEY'] = groq_api
#os.environ['OPENAI_API_KEY'] = openai_api_key
llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=openai_api_key)
llm2 = ChatGroq(temperature=0, model="mixtral-8x7b-32768", api_key=groq_api)



# Create the CSV agent
agent = create_csv_agent(llm2, csv_file_path, verbose=True, allow_dangerous_code=True)

def query_data(query):
    response = agent.invoke(query)
    return response

#query = "how many rows are there?"
query = input("please enter your query:")
response = query_data(query)
print(response)
