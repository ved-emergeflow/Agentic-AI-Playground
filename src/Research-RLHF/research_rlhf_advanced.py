# ----------------------------------------------- Imports ---------------------------------------------------

import os
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import tool
from langgraph.types import Command, interrupt
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_community.utilities import SearxSearchWrapper, DuckDuckGoSearchAPIWrapper
from pydantic import BaseModel, Field

# ------------------------------------------------ State -------------------------------------------------------

import operator


class SearchAgentState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]
    product: str
    region: str
    max_results: int
    competitors: list[str]
    product_results: str
    competitor_results: list[str]
    summary: str

class ResearchState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]
    first_run: bool
    search_results: list
    first_research: list[str]
    current_research: list[str]
    final_research: list[str]
    changes: list[str]
    approved: bool
    insights: str


class MasterState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]
    product: str
    competitors: list[str]
    search_results: Annotated[list, operator.add]
    first_run: bool
    approved: bool
    research_summary: str
    insights: str


# --------------------------------------------------------- Model -------------------------------------------

MODEL = 'meta-llama/llama-4-maverick-17b-128e-instruct'

def get_groq_model(model):

    return ChatGroq(
        model=model,
        temperature=0.2,
        max_retries=2,
        api_key=os.getenv('GROQ_API_KEY')
    )

model = get_groq_model(MODEL)

# ------------------------------------------------------ Output Parsers ----------------------------------------

class Research(BaseModel):
    product_target_age_group: list[Literal['Gen Alpha (18 and below)',
                                            'Gen Z (18-30)',
                                            'Millennial (30-40)', # type: ignore
                                            'Gen X (40-60)',
                                            'Boomer (60 and above)', # type: ignore
                                            'Everyone']] = Field(description="Target Age Group for the product")
    product_target_gender: Literal['Male', 'Female', 'Non-binary', 'Everyone'] = Field(description="Target Gender for the product")
    product_target_persona: str = Field(description='Target group who would be interested in the product')
    competitor_target_age_group: list[Literal['Gen Alpha (18 and below)',
                                            'Gen Z (18-30)',
                                            'Millennial (30-40)',
                                            'Gen X (40-60)',
                                            'Boomer (60 and above)',
                                            'Everyone']] = Field(description="Target Age Group for the competitors")
    competitor_target_gender: Literal['Male', 'Female', 'Non-binary', 'Everyone'] = Field(description="Target Gender for the competitors")
    competitor_target_persona: str = Field(description='Target group who would be interested in the competitors')

research_parser = JsonOutputParser(pydantic_object=Research)

# ------------------------------------------------------- Tools ------------------------------------------------------

wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
search_ddgs = DuckDuckGoSearchRun(api_wrapper=wrapper)
search_ddgs_results = DuckDuckGoSearchResults(api_wrapper=wrapper)

# search = SearxSearchWrapper(searx_host="http://localhost:8888")

@tool('search_product')
def search_product(site: str, product: str=''):
    """
    This tool searches the provided website for product information
    and/or directly searches the product webpage for information.
    :param site: Name of the website or product website page
    :param product: Name of the product (Optional)
    :return: str
    """

    results = search_ddgs.invoke(f'{site}')

    return results

@tool('search_competitors')
def search_competitors(sites: list[str], competitors=None):
    """
    This tool searches the provided website for competitor product information
    and/or directly searches the product webpage for information.
    :param sites: List of the websites or competitor product website pages
    :param competitors: List of the names of the competitor products (Optional)
    :return: list[str]
    """
    results = []
    if competitors is None:
        competitors = ['']*len(sites)

    for site, competitor in zip(sites,competitors):
        result = search_ddgs.invoke(f'{site} {competitor}')
        results.append(result)

    return results

@tool('deep_search_product')
def deep_search_product(site:str,
                         engines:list[str]=['google','brave','duckduckgo'],
                         time_range:str='year',
                         product:str='',
                         num_results:int = 3):
    """
    This tool searches the provided website for product information based on the additional parameters
    :param site: Name of the website or product website page
    :param engines: List of the names of the engines (Optional)
    :param time_range: Time range of the search (Optional)
    :param product: Name of the product (Optional)
    :param num_results: Number of results (Optional)
    :return: list[dict]
    """

    results = search_ddgs_results.invoke(
        f'{site} {product}',
        time_range=time_range,
        engines=engines,
        num_results=num_results
    )

    return results


@tool('deep_search_competitors')
def deep_search_competitors(sites:list[str],
                         engines:list[str]=['google','brave','duckduckgo'],
                         time_range:str='year',
                         competitors:list[str]=None,
                         num_results:int = 3):
    """
    This tool searches the provided website for product information based on the additional parameters
    :param sites: List of the names of the websites or competitor product website pages
    :param engines: List of the names of the engines (Optional)
    :param time_range: Time range of the search (Optional)
    :param competitors: List of the names of the competitor products (Optional)
    :param num_results: Number of results (Optional)
    :return: list[dict]
    """

    all_results = []

    if competitors is None:
        competitors = ['']*len(sites)

    for site,competitor in zip(sites,competitors):
        results = search_ddgs_results.invoke(
            f'{site} {competitor}',
            time_range=time_range,
            engines=engines,
            num_results=num_results
        )
        all_results.append(results)

    return all_results

search_tools = [search_product, search_competitors]
# search_tools_deep = [search_product, search_competitors, deep_search_product, deep_search_competitors]

# ------------------------------------------------------- Search Agent Nodes --------------------------------------

def call_llm(state: SearchAgentState):
    print('\nEntering Model.....')
    messages = state['messages']
    res = model.bind_tools(search_tools).invoke(messages)
    if isinstance(res, AIMessage):
        res = [res]
    print([(tool['name'], tool['args']) for tool in res[-1].tool_calls])
    if len(res) > 3:

        if len(res[-1].tool_calls) == 0:

            return {
                'messages': res,
                'summary': res[-1].content
            }

        else:

            return {
                'messages': res
            }

    else:
        return {
            'messages': res

        }

# Create ToolNode with retry logic
# def create_tool_node_with_retry(tools, max_retries=3):
#     """Wrapper to add retry logic to ToolNode"""
#     tool_node = ToolNode(tools)
#
#     def tool_node_with_retry(state: MessagesState):
#         for attempt in range(max_retries):
#             result = tool_node.invoke(state)
#
#             # Check if any tool returned "No good search result found"
#             messages = result.get("messages", [])
#             if messages and isinstance(messages[-1], ToolMessage):
#                 if messages[-1].content != 'No good search result found':
#                     return result
#                 print(f"Retry {attempt + 1}/{max_retries}")
#
#         # After all retries, return failure message
#         return {
#             "messages": [
#                 ToolMessage(
#                     content="Failed after retries: No good search result found",
#                     tool_call_id=state["messages"][-1].tool_calls[0]["id"]
#                 )
#             ]
#         }
#
#     return tool_node_with_retry

def run_tools(state: SearchAgentState):
    print('\nEntering Tool Node.....')
    last_message = state['messages'][-1]
    product = state['product']
    competitors = state['competitors']
    if len(last_message.tool_calls)>0 and isinstance(last_message, AIMessage):

        tool_messages = []
        product_results = []
        competitor_results = []
        tool_calls = last_message.tool_calls
        names = [tool['name'] for tool in tool_calls]

        for tool in tool_calls:
            tool_call_id = tool['id']
            if tool['name'] in names:

                current_tool = [t for t in search_tools if t.name == tool['name']][0]
                if current_tool.name in ['search_product', 'deep_search_product']:
                    res = current_tool.invoke({
                        'site': product
                    })
                    tool_messages.append(ToolMessage(content=res, tool_call_id=tool_call_id))
                elif current_tool.name in ['search_competitors','deep_search_competitors']:
                    res = current_tool.invoke({
                        'sites': competitors
                    })
                    tool_messages.append(ToolMessage(content='\n\n'.join(res), tool_call_id=tool_call_id))

                if current_tool.name in ['search_product', 'deep_search_product']:
                    product_results.append(res)
                elif current_tool.name in ['search_competitors','deep_search_competitors']:
                    competitor_results.append(res)

            else:
                tool_messages.append(ToolMessage(content='Invalid tool called', tool_call_id=tool_call_id))

        print('\n',product_results)
        print('\n',competitor_results)
        return {
            'messages': tool_messages,
            'product_results': ' '.join(product_results) if len(product_results) > 0 else '',
            'competitor_results': ' '.join(competitor_results[0]) if len(competitor_results[0]) > 0 else '',
        }

def is_tool_call(state: SearchAgentState)->Literal[True, False]:
    print('\nEntering Tool condition.....')
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage):

        if len(last_message.tool_calls)>0:
            return True
        else:
            return False
    else:
        return False

def handoff(state: SearchAgentState):
    print('\nEntering Handoff.....')
    if 'product_results' in state:
        print('\nHandoff to Research Agent')
        return Command(
            update={
                'messages' : state['messages'],
                'search_results' : [state['product_results'],state['competitor_results']],
                'first_run': True,
                'approved': False
            },
            goto='research',
            graph=Command.PARENT

        )
    else:
        return Command(
            goto='call_llm'
        )
    
# ---------------------------------------------------- Search Agent Graph -----------------------------------
    
search_graph = StateGraph(SearchAgentState)
search_graph.add_node('call_llm', call_llm)
search_graph.add_node('run_tools', run_tools)
search_graph.add_node('handoff', handoff)

search_graph.add_edge(START, 'call_llm')
search_graph.add_conditional_edges('call_llm', is_tool_call, {
    True: 'run_tools',
    False: 'handoff'
})
search_graph.add_edge('run_tools', 'call_llm')

# ---------------------------------------------------- Research Agent Node ------------------------------------------

def create_research(state: ResearchState):
    print('\nCreating Research.....')

    messages = [SystemMessage(content=f"""
                    You are a researcher Agent. Your task is to create a research report on the given product information and competitors to figure out target age groups, genders and personas. 
                              """),
                HumanMessage(content=f"""
                    Create a Research Report on the following product: {state['search_results'][0]} and it's competitors : {state['search_results'][1]}. 
                    Also incorporate changes by the human: {state['changes'] if 'changes' in state else 'No changes'}.
                    Return response in JSON format using following format: {research_parser.get_format_instructions()}""")]

    chain = model.bind(response_format={'type':'json_object'}) | research_parser
    res = chain.invoke(messages)

    outmessage = f"""
        The created research is as follows:
        Product:
        Target Age Group : {res['product_target_age_group']}
        Target Gender : {res['product_target_gender']}
        Target Persona : {res['product_target_persona']}
        Competitors:
        Target Age Group : {res['competitor_target_age_group']}
        Target Gender : {res['competitor_target_gender']}
        Target Persona : {res['competitor_target_persona']}
"""

    product_research = [res['product_target_age_group'], res['product_target_gender'], res['product_target_persona']]
    competitor_research = [res['competitor_target_age_group'], res['competitor_target_gender'], res['competitor_target_persona']]

    return {
        'messages': AIMessage(content=outmessage),
        'first_run': False,
        'first_research': [product_research, competitor_research] if state['first_run'] else state['first_research'],
        'current_research': [product_research, competitor_research]
    }

def hitl(state: ResearchState):

    def safe_append_changes(state: dict, new_change):
        if 'changes' not in state or state['changes'] is None:
            state['changes'] = [new_change]
            return
        if isinstance(state['changes'], list):
            state['changes'].append(new_change)
            return

        state['changes'] = [state['changes'], new_change]


    print("\nEntering HITL.....")
    raw = interrupt(state['current_research'])

    if isinstance(raw, dict):
        new_change = raw.get('changes') or raw.get('message') or json.dumps(raw)
        if isinstance(new_change, list):
            new_change = ", ".join(map(str, new_change))
    else:
        new_change = str(raw)

    if new_change.strip().lower() in ('approved', 'okay', 'ok', 'yes'):
        safe_append_changes(state, 'No changes')
        return {
            'final_research': state['current_research'],
            'changes': state['changes'],
            'approved': True
        }

    safe_append_changes(state, new_change)
    return {
        'approved': False,
        'changes': state['changes']
    }



def is_approved(state: ResearchState)->Literal[True, False]:
    print('\nEntering Approved condition.....')

    if state['approved']:
        return True
    else:
        return False

def create_insights(state: ResearchState):
    print('\nCreating Insights...')

    message = HumanMessage(content=f"""
                Create an insights report by analyzing the first research report: {state['first_research']}, the final research report: {state['final_research']} and all the human made changes {state['changes']} in order to identify patterns and use them the next run.
""")

    res = model.invoke([message])

    return Command(
        graph=Command.PARENT,
        update={
        'messages': res,
        'research_summary': state['final_research'],
        'insights': res.content
    })

# -------------------------------------------------- Research Agent Graph -----------------------------

research_graph = StateGraph(ResearchState)
research_graph.add_node('create_research', create_research)
research_graph.add_node('hitl', hitl)
research_graph.add_node('create_insights', create_insights)
research_graph.add_edge(START, 'create_research')
research_graph.add_edge('create_research', 'hitl')
research_graph.add_conditional_edges('hitl', is_approved, {
    True: 'create_insights',
    False: 'create_research'
})

# ------------------------------------------------------ Subagent Utils ------------------------------------

def get_search_agent(checkpointer):

    return search_graph.compile(checkpointer=checkpointer)

def get_research_agent(checkpointer):

    return research_graph.compile(checkpointer=checkpointer)

# -------------------------------------------------------- Master Graph --------------------------------------

def get_advanced_research_agent(checkpointer):

    master_graph = StateGraph(MasterState)
    master_graph.add_node('search', get_search_agent(checkpointer=checkpointer))
    master_graph.add_node('research', get_research_agent(checkpointer=checkpointer))
    master_graph.add_edge(START, 'search')

    return master_graph.compile(checkpointer=checkpointer)