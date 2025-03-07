import os
from typing import Dict, List, Tuple, Any, Optional
from pydantic import BaseModel, Field
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables (still useful as fallback)
load_dotenv()

# Configure page
st.set_page_config(page_title="AI Blog Generator", layout="wide")

# API Key handling in sidebar
with st.sidebar:
    st.title("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key here")
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.divider()
    st.write("## About")
    st.write("This app uses LangGraph to generate structured blog posts with a multi-step workflow.")
    st.write("Made with ❤️ using LangGraph and Streamlit")

# Define the state schema
class BlogGeneratorState(BaseModel):
    topic: str = Field(default="")
    audience: str = Field(default="")
    tone: str = Field(default="")
    word_count: int = Field(default=500)
    outline: List[str] = Field(default_factory=list)
    sections: Dict[str, str] = Field(default_factory=dict)
    final_blog: str = Field(default="")
    error: Optional[str] = Field(default=None)

# Initialize LLM (only when API key is available)
def get_llm():
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please enter your OpenAI API key in the sidebar")
        st.stop()
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Create prompt templates
outline_prompt = ChatPromptTemplate.from_template(
    """You are a professional blog writer. Create an outline for a blog post about {topic}.
    The audience is {audience} and the tone should be {tone}.
    The blog should be approximately {word_count} words.
    
    Return ONLY the outline as a list of section headings (without numbers or bullets).
    Each heading should be concise and engaging."""
)

section_prompt = ChatPromptTemplate.from_template(
    """Write content for the following section of a blog post about {topic}:
    
    Section: {section}
    
    The audience is {audience} and the tone should be {tone}.
    Make this section approximately {section_word_count} words.
    Make the content engaging, informative, and valuable to the reader.
    
    Return ONLY the content for this section, without the heading."""
)

final_assembly_prompt = ChatPromptTemplate.from_template(
    """You have a blog post with the following sections:
    
    {sections_content}
    
    Format this into a complete, professional blog post in Markdown format with:
    1. An engaging title at the top as an H1 heading
    2. A brief introduction before the first section
    3. Each section heading as an H2
    4. A conclusion at the end
    5. Proper spacing between sections
    6. 2-3 relevant markdown formatting elements like bold, italic, blockquotes, or bullet points where appropriate
    
    The blog should maintain the {tone} tone and be targeted at {audience}.
    Make it flow naturally between sections."""
)

# Define the nodes for the graph
def get_outline(state: BlogGeneratorState) -> BlogGeneratorState:
    """Generate an outline for the blog post."""
    try:
        llm = get_llm()
        chain = outline_prompt | llm
        response = chain.invoke({
            "topic": state.topic,
            "audience": state.audience, 
            "tone": state.tone,
            "word_count": state.word_count
        })
        
        # Parse the outline into a list
        output_text = response.content
        outline = [line.strip() for line in output_text.split('\n') if line.strip()]
        return BlogGeneratorState(**{**state.model_dump(), "outline": outline})
    except Exception as e:
        st.error(f"Outline Error: {str(e)}")
        st.write(f"Response type: {type(response)}")
        if hasattr(response, '__dict__'):
            st.write(f"Response attributes: {response.__dict__}")
        return BlogGeneratorState(**{**state.model_dump(), "error": f"Error generating outline: {str(e)}"})

def generate_sections(state: BlogGeneratorState) -> BlogGeneratorState:
    """Generate content for each section in the outline."""
    if state.error:
        return state
    
    sections = {}
    section_word_count = state.word_count // len(state.outline)
    
    try:
        llm = get_llm()
        chain = section_prompt | llm
        
        for section in state.outline:
            response = chain.invoke({
                "topic": state.topic,
                "section": section,
                "audience": state.audience,
                "tone": state.tone,
                "section_word_count": section_word_count
            })
            
            sections[section] = response.content
            
        return BlogGeneratorState(**{**state.model_dump(), "sections": sections})
    except Exception as e:
        return BlogGeneratorState(**{**state.model_dump(), "error": f"Error generating sections: {str(e)}"})

def assemble_blog(state: BlogGeneratorState) -> BlogGeneratorState:
    """Assemble the final blog post in Markdown format."""
    if state.error:
        return state
    
    try:
        llm = get_llm()
        chain = final_assembly_prompt | llm
        
        sections_content = "\n\n".join([f"Section: {heading}\nContent: {content}" 
                                   for heading, content in state.sections.items()])
        
        response = chain.invoke({
            "sections_content": sections_content,
            "tone": state.tone,
            "audience": state.audience
        })
        
        final_blog = response.content
        return BlogGeneratorState(**{**state.model_dump(), "final_blog": final_blog})
    except Exception as e:
        return BlogGeneratorState(**{**state.model_dump(), "error": f"Error assembling blog: {str(e)}"})

# Define the workflow graph
def create_blog_generator_graph():
    workflow = StateGraph(BlogGeneratorState)
    
    # Add nodes
    workflow.add_node("get_outline", get_outline)
    workflow.add_node("generate_sections", generate_sections)
    workflow.add_node("assemble_blog", assemble_blog)
    
    # Add edges
    workflow.add_edge("get_outline", "generate_sections")
    workflow.add_edge("generate_sections", "assemble_blog")
    workflow.add_edge("assemble_blog", END)
    
    # Set the entry point
    workflow.set_entry_point("get_outline")
    
    return workflow.compile()

# Create the Streamlit app main content
st.title("AI Blog Generator")
st.write("Generate professional blog posts with a structured workflow")

with st.form("blog_generator_form"):
    topic = st.text_input("Blog Topic", placeholder="E.g., Sustainable Living in Urban Environments")
    
    col1, col2 = st.columns(2)
    with col1:
        audience = st.text_input("Target Audience", placeholder="E.g., Young professionals")
        tone = st.selectbox("Tone", ["Informative", "Conversational", "Professional", "Inspirational", "Technical"])
    
    with col2:
        word_count = st.slider("Approximate Word Count", min_value=300, max_value=2000, value=800, step=100)
    
    submit_button = st.form_submit_button("Generate Blog")

if submit_button:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please enter your OpenAI API key in the sidebar before generating a blog")
    elif not topic or not audience:
        st.error("Please fill out all required fields.")
    else:
        with st.spinner("Generating your blog post..."):
            try:
                # Initialize the graph
                blog_generator = create_blog_generator_graph()
                
                # Set the initial state
                initial_state = BlogGeneratorState(
                    topic=topic,
                    audience=audience,
                    tone=tone,
                    word_count=word_count
                )
                
                # Run the graph
                result = blog_generator.invoke(initial_state)
                
                # Check if result is a dict and has expected keys
                if isinstance(result, dict):
                    final_blog = result.get("final_blog", "")
                    outline = result.get("outline", [])
                    error = result.get("error")
                    
                    if error:
                        st.error(f"Error: {error}")
                    elif final_blog:
                        # Display the blog post
                        st.success("Blog post generated successfully!")
                        
                        st.subheader("Generated Blog Post")
                        st.markdown(final_blog)
                        
                        # Download button for the blog post
                        st.download_button(
                            label="Download Blog as Markdown",
                            data=final_blog,
                            file_name=f"{topic.replace(' ', '_').lower()}_blog.md",
                            mime="text/markdown",
                        )
                        
                        # Optionally show the outline
                        with st.expander("View Blog Outline"):
                            for i, section in enumerate(outline, 1):
                                st.write(f"{i}. {section}")
                    else:
                        st.error("Blog generation completed but no content was produced")
                else:
                    st.error(f"Unexpected result type: {type(result)}")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please check your API key and try again.")