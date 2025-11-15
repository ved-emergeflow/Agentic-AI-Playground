import uuid
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
# from research_rlhf_agent_groq import get_groq_agent
from research_rlhf_advanced import get_advanced_research_agent
import streamlit as st
from datetime import datetime
import json

st.set_page_config(
    page_title="Agent",
    page_icon="💬",
    layout="wide"
)

checkpointer = InMemorySaver()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "agent" not in st.session_state:
    st.session_state.agent = get_advanced_research_agent(checkpointer)

config = {
    'configurable': {
        'thread_id': st.session_state.thread_id
    }
}

st.title("💬 Chat Interface")

# --- session state defaults ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "insights" not in st.session_state:
    st.session_state.insights = ""

if "interrupted" not in st.session_state:
    st.session_state.interrupted = False

if "interrupt_response" not in st.session_state:
    st.session_state.interrupt_response = None

if "interrupt_value" not in st.session_state:
    st.session_state.interrupt_value = None

# Add state for product & competitors inputs (optional defaults)
if "product_input" not in st.session_state:
    st.session_state.product_input = ""

if "competitors_input" not in st.session_state:
    st.session_state.competitors_input = ""  # comma-separated string

# --- helper functions ---
def get_last_message_with_content(messages_list):
    """Find the last message with actual content"""
    for msg in reversed(messages_list):
        if hasattr(msg, 'content') and msg.content and msg.content.strip():
            return msg.content
    return None

def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- input area: two distinct inputs for product and competitors ---
with st.sidebar:
    st.header("Inputs")
    st.text_input("Product name", key="product_input", placeholder="Enter product name (e.g., Widget X)")
    st.text_input("Competitors (comma-separated)", key="competitors_input", placeholder="Competitor A, Competitor B")

display_messages()

# Handle HITL input (unchanged)
if st.session_state.interrupted and st.session_state.interrupt_response:
    # Display the agent's response that caused interrupt
    with st.chat_message("assistant"):
        st.markdown(st.session_state.interrupt_response)

    st.info("⏸️ Agent paused - provide feedback to continue")
    st.write("**Current Analysis:**")
    if st.session_state.interrupt_value:
        if isinstance(st.session_state.interrupt_value, dict):
            st.json(st.session_state.interrupt_value)
        else:
            st.write(st.session_state.interrupt_value[0].value)

    hitl_input = st.chat_input("Your feedback (or type 'Approved' to continue)...")

    if hitl_input:
        # Add human feedback to messages
        st.session_state.messages.append({
            "role": "user",
            "content": hitl_input,
            "timestamp": datetime.now()
        })

        with st.chat_message("user"):
            st.markdown(hitl_input)

        with st.chat_message("assistant"):
            with st.spinner("Agent processing feedback..."):
                try:
                    # Resume with same agent instance
                    response = st.session_state.agent.invoke(
                        Command(resume=hitl_input),
                        config=config
                    )

                    st.session_state.interrupted = False
                    st.session_state.interrupt_response = None
                    st.session_state.interrupt_value = None

                    # Check if still interrupted
                    if response.get("__interrupt__"):
                        # Still interrupted after resume
                        final_response = get_last_message_with_content(response['messages'])
                        if final_response:
                            st.session_state.interrupt_response = final_response
                            st.session_state.interrupted = True
                            st.session_state.interrupt_value = response.get("__interrupt__")
                        st.rerun()
                    else:
                        # Not interrupted anymore - completed
                        final_response = get_last_message_with_content(response['messages'])
                        if final_response:
                            st.markdown(final_response)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": final_response,
                                "timestamp": datetime.now()
                            })
                        st.markdown(f"\nInsights gained: {response.get('insights')}")
                        st.success("✅ Research completed!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

# Main chat input (user message). We pull product & competitors from sidebar inputs.
elif user_msg := st.chat_input("Type your message..."):
    # Read product & competitors from session_state (set via sidebar text_inputs)
    product = st.session_state.get("product_input", "").strip()
    competitors_raw = st.session_state.get("competitors_input", "").strip()
    # convert comma-separated competitors to list (clean)
    competitors = [c.strip() for c in competitors_raw.split(",") if c.strip()] if competitors_raw else []

    # Save the chat message and also include product/competitors in what the agent gets
    st.session_state.messages.append({
        "role": "user",
        "content": f"Message: {user_msg}\nProduct: {product}\nCompetitors: {', '.join(competitors)}",
        "timestamp": datetime.now()
    })

    with st.chat_message("user"):
        # show what the user sent along with product/competitors for clarity
        st.markdown(f"**Product:** {product}\n\n**Competitors:** {', '.join(competitors)}\n\n**Message:** {user_msg}")

    with st.chat_message("assistant"):
        with st.spinner("Agent thinking..."):
            init_state = {
                'messages': [HumanMessage(content=f'Perform web search on the product: {product} and their competitors: {competitors}')],
                'product': product,
                'competitors': competitors
            }

            try:
                response = st.session_state.agent.invoke(init_state, config=config)

                # Check if interrupted
                if response.get("__interrupt__"):
                    interrupt_msg = get_last_message_with_content(response['messages'])
                    interrupt_val = response.get("__interrupt__")

                    if interrupt_msg:
                        print(f'\n✅ Interrupt Found: {interrupt_msg}')
                        print(f'Interrupt Value: {interrupt_val}')

                        st.session_state.interrupt_response = interrupt_msg
                        st.session_state.interrupt_value = interrupt_val
                        st.session_state.interrupted = True
                        st.markdown(interrupt_msg)
                        st.warning("⏸️ Waiting for your input to continue...")
                    else:
                        st.warning("⚠️ Interrupt triggered but no message content found")

                    st.rerun()
                else:
                    # Not interrupted - completed normally
                    final_response = get_last_message_with_content(response['messages'])
                    if final_response:
                        st.markdown(final_response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_response,
                            "timestamp": datetime.now()
                        })
                    st.markdown(f"\nInsights gained: {response.get('insights')}")
                    st.success("✅ Research completed!")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
