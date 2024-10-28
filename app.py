import getpass
import os
import ast
from urllib.parse import quote_plus
from sqlalchemy.engine import create_engine
from langchain_community.utilities import SQLDatabase
import oracledb
import gradio as gr
import pandas as pd
from sqlalchemy import text
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_openai import ChatOpenAI
OPENAI_API_KEY = "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"

def query_to_dataframe(query_result, db, query):
    """Convert database query result to pandas DataFrame with column names."""
    with db._engine.connect() as connection:
        sql = text(query)
        result = connection.execute(sql)
        column_names = result.keys()
    
    if isinstance(query_result, str):
        clean_result = query_result.replace('datetime.datetime', '')
        try:
            data = ast.literal_eval(clean_result)
        except:
            data = eval(clean_result)
    else:
        data = query_result
    
    return pd.DataFrame(data, columns=column_names)

# Database setup
username = "SYSTEM"
password = "Passw0rd@321"
host = "localhost"
port = "1521"
service = "XEPDB1"

dsn = f"""(DESCRIPTION=
    (ADDRESS=(PROTOCOL=TCP)(HOST={host})(PORT={port}))
    (CONNECT_DATA=(SERVICE_NAME={service})))"""

encoded_password = quote_plus(password)
uri = f"oracle+oracledb://{username}:{encoded_password}@"

engine = create_engine(
    uri,
    connect_args={"dsn": dsn}
)

include_tables = ['dw_boroughs_new1', 'dw_incidents1']
db = SQLDatabase(engine, include_tables=include_tables)

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-4o-mini")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
handler = StdOutCallbackHandler()

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
system_message = prompt_template.format(dialect="oracle", top_k=5)

agent_executor = create_react_agent(
    llm, 
    toolkit.get_tools(), 
    state_modifier=system_message
)

def process_query(user_query):
    try:
        # Execute query with callback handler
        result = agent_executor.invoke(
            {"messages": [("user", user_query)]},
            config={"callbacks": [handler]}
        )
        print("result:",result)
        print("final result check:",result["messages"][-1])
        # Extract metadata and content
        response_metadata = result["messages"][-1].response_metadata["token_usage"]
        usage_metadata = result["messages"][-1].usage_metadata
        nl_output = result["messages"][-1].content
        
        # Extract SQL query from tool call
        tool_call = result["messages"][-3].additional_kwargs
        arg = tool_call['tool_calls'][0]["function"]["arguments"]
        sql_query = ast.literal_eval(arg)["query"]
        print("sql_query:",sql_query)
        # Execute SQL query and get DataFrame
        query_result = db.run(sql_query)
        final_df = query_to_dataframe(query_result, db, sql_query)
        print("final_df:",final_df)
        # Format metadata for display
        metadata_str = (
            f"Token Usage for usage metadata:\n"
            f"Input tokens: {usage_metadata['input_tokens']}\n"
            f"Output tokens: {usage_metadata['output_tokens']}\n"
            f"Total tokens: {usage_metadata['total_tokens']}\n"
            f"Token Usage for response metadata:\n"
            f"Completion tokens: {response_metadata['completion_tokens']}\n"
            f"Prompt tokens: {response_metadata['prompt_tokens']}\n"
            f"Total tokens: {response_metadata['total_tokens']}\n\n"
            f"SQL Query:\n{sql_query}\n\n"
            f"Natural Language Response:\n{nl_output}"
        )
        print("metadata_str:",metadata_str)
        return final_df, metadata_str, sql_query, nl_output
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return None, error_msg, "", ""

def create_interface():
    with gr.Blocks(title="Advanced SQL Query Assistant") as interface:
        gr.Markdown("""
        # Advanced SQL Query Assistant
        Ask questions about the incidents and boroughs data in natural language.
        Get detailed information about query execution and results.
        """)
        
        with gr.Row():
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="Example: What are the codes and descriptions for each borough?",
                lines=3
            )
        
        with gr.Row():
            submit_btn = gr.Button("Submit Query", variant="primary")
        
        with gr.Row():
            with gr.Column():
                output_table = gr.Dataframe(label="Query Results")
            
        with gr.Row():
            with gr.Column():
                metadata_output = gr.Textbox(
                    label="Query Details",
                    lines=10,
                    interactive=False
                )
                sql_output = gr.Code(
                    label="SQL Query",
                    language="sql",
                    interactive=False
                )
                nl_output = gr.Textbox(
                    label="Natural Language Response",
                    lines=5,
                    interactive=False
                )
        
        gr.Examples(
            examples=[
                "What are the codes and descriptions for each borough?",
                "How many incidents were reported in each borough?",
                "Show me the total number of incidents by month",
            ],
            inputs=query_input,
        )
        
        submit_btn.click(
            fn=process_query,
            inputs=[query_input],
            outputs=[output_table, metadata_output, sql_output, nl_output]
        )
    
    return interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
