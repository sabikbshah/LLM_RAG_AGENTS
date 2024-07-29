#LLM_RAG-Agent
the project has 4 different independent scripts.
Note: the stocklive2.csv is already present. Database file is popular chinook.db

install both requirements.txt pip install -r requirement.txt requirements2.txt

1.RUN the **finalcsv.py** > only uses agent

2.**final.py** > uses chathistory with agent has somedependency issues but may run in other computer. other two files.

3.**chatmysql2.ipynb** > without AGENTS Uses langchain create-sqlquery-chain. create-sql-query-chain instance provdes > Prompts template already preused also customizable in Runnable PRECAUTION the whole database would be loaded high amount of token will be used!!!

4.**chatmysqlagent.py** > uses agent and Frontend streamlitapp. to run >> py streamlit chatmysqlagent.py PRECAUTION same as above
