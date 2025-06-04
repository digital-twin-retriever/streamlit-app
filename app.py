import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import random
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from google.api_core import retry


st.set_page_config(page_title="Digital Twin Retriever", page_icon=":robot_face:")

# Configure API key
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
  genai.models.Models.generate_content = retry.Retry(
      predicate=is_retriable)(genai.models.Models.generate_content)

# Load data
@st.cache_resource
def load_case_data():
    case_df = pd.read_parquet("https://zenodo.org/records/15310586/files/case_texts.parquet?download=1")
    return case_df

@st.cache_resource
def load_embedding_data():
    emb_df = pd.read_parquet("https://zenodo.org/records/15310586/files/case_embeddings.parquet?download=1")
    return emb_df

with st.spinner("📄 Loading clinical case texts..."):
    case_df = load_case_data()

with st.spinner("🧠 Loading case embeddings..."):
    emb_df = load_embedding_data()

# Global CSS 
st.markdown(
    """
    <style>

    div.stExpander + div.stElementContainer .stMarkdown {
        padding: 1rem;
    }

    a {
        text-decoration: none !important;
        color: #6172e0 !important;
    }

    .st-emotion-cache-1d2o6qs {
        max-width: 1000px !important;
    }

    header {
        background: transparent !important;
    }

       
    div[data-testid="stSidebarUserContent"] {
        padding: 0px 1.5rem 2rem;   
    }

    .stLogo {
        margin: 1rem auto;
        height: 6vw;
        min-height: 60px;
        max-height: 100px;
    }

    .stSidebar {
        border-right: 2px solid #dfe1ea;
        background: #F5F5FA !important;
    }   

    .stSidebar p {
        text-align: center !important;
    }

    .stSidebar p:nth-of-type(2) {
        text-align: center !important;
        color: #757a8e;
        font-size: 14px;
    }

    .stMain {
        background-image: url('https://raw.githubusercontent.com/digital-twin-retriever/streamlit-app/refs/heads/main/img/wave-bg.webp');
        background-repeat: no-repeat;
        background-position: top right;
        background-size: 100%;
        background-color: rgba(255, 255, 255, 0.5); 
        background-blend-mode: overlay;
    }

    .st-emotion-cache-1c7y2kd {
        background-color: #F5F5FA;
        padding-right: 2rem;
    }

    .stMain button, .stSidebar button {
        margin: auto;
        display: block;
    }

    .st-emotion-cache-bho8sy {
        background-color: #6172e0;
    }

    .custom-container h1 {
        text-align: center !important;
        font-size: xx-large !important;
        color: #38386A !important;
        padding-top: 0 !important;
        margin: 0 !important;
        padding-bottom: 1rem !important;
    }

    .custom-container p {
        text-align: center !important;
        margin-bottom: 0 !important;
    }

    .custom-container img {
        max-height: 18vh !important;
        width: auto !important;
        display: block;
        margin: 1rem auto 0 auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)






# Session State variables
if "similar_cases_df" not in st.session_state:
    st.session_state.similar_cases_df = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False


# Model functions
def find_top_similar(query: str, top_k: int = 10, similarity_threshold: float = 0.68) -> pd.Series:
    """Return top_k most semantically similar cases above the threshold."""
    norm = np.linalg.norm(emb_df.values, axis=1, keepdims=True)
    normed_embeddings = emb_df.div(norm, axis=0)
    query_vector = np.array(
        client.models.embed_content(
            model="models/text-embedding-004",
            contents=[query],
            config=types.EmbedContentConfig(task_type="semantic_similarity")
        ).embeddings[0].values
    )
    query_vector /= np.linalg.norm(query_vector)
    scores = normed_embeddings.values @ query_vector
    result = pd.Series(scores, index=emb_df.index)
    return result[result >= similarity_threshold].nlargest(top_k)


def fetch_citation(pmcid: str) -> str:
    """Fetch APA-style citation for a given PMCID."""
    try:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=PMCID:{pmcid}&format=json"
        result = requests.get(url, timeout=10).json()["resultList"]["result"][0]
        return f"{result.get('authorString', 'Unknown authors')} ({result.get('pubYear', 'n.d.')}). " \
               f"{result.get('title', 'No title')} *{result.get('journalTitle', 'Unknown journal')}*, " \
               f"{result.get('journalVolume', '')}({result.get('issue', '')}), {result.get('pageInfo', '')}. " \
               f"https://doi.org/{result.get('doi', '')}"
    except Exception as e:
        return f"Error fetching citation for {pmcid}: {str(e)}"


def fetch_discussion(pmcid: str) -> str:
    """Return discussion section from a PMC article."""
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmcid}/ascii"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return f"Error fetching discussion for {pmcid}"
        soup = BeautifulSoup(response.content, "xml")
        discussion = [
            p.find("text").get_text(strip=True)
            for p in soup.find_all("passage")
            if (
                p.find("infon", {"key": "section_type"}, string=re.compile("DISCUSS", re.I))
                and p.find("infon", {"key": "type"}, string="paragraph")
                and p.find("text")
            )
        ]
        if discussion:
            return " ".join(discussion)
        passages = soup.find_all("passage")
        for i, p in enumerate(passages):
            if (
                p.find("infon", {"key": "type"}, string=re.compile("title", re.I))
                and p.find("text", string=re.compile("discuss", re.I))
            ):
                discussion = [
                    np.find("text").get_text(strip=True)
                    for np in passages[i + 1 :]
                    if np.find("infon", {"key": "type"}, string="paragraph") and np.find("text")
                ]
                break
        return " ".join(discussion) if discussion else "Discussion not found."
    except Exception as e:
        return f"Error processing discussion for {pmcid}: {e}"


def get_case_data(pmcids: list[str], max_workers: int = 10) -> dict:
    """Parallel fetch of citation and discussion info for PMCIDs."""
    def fetch(pmcid):
        return pmcid, fetch_citation(pmcid), fetch_discussion(pmcid)
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for future in as_completed([executor.submit(fetch, pid) for pid in pmcids]):
            try:
                pmcid, citation, discussion = future.result()
                results[pmcid] = {"citation": citation, "discussion": discussion}
            except Exception as e:
                results[pmcid] = {"citation": f"Error: {e}", "discussion": ""}
    return results


def compile_similar_cases(input_query: str, case_df: pd.DataFrame = case_df, emb_df: pd.DataFrame = emb_df) -> pd.DataFrame:
    """Find and enrich similar clinical cases with citations and discussions."""
    top_similar = find_top_similar(input_query)
    df = case_df.loc[case_df['case_id'].isin(top_similar.index)].copy()
    df["case_similarity_score"] = df["case_id"].map(top_similar)
    df.sort_values(by="case_similarity_score", ascending=False, inplace=True)
    pmcids = pd.Series(top_similar.index).str.extract(r"(PMC\d+)")[0].dropna().unique()
    article_info = get_case_data(pmcids)
    df["citation"] = df["article_id"].map(lambda x: article_info.get(x, {}).get("citation", "N/A"))
    df["discussion"] = df["article_id"].map(lambda x: article_info.get(x, {}).get("discussion", "N/A"))
    return df


def find_cases(query: str) -> str:
    """Retrieve and cache similar clinical cases for a query."""
    if st.session_state.get("last_query") == query and st.session_state.get("similar_cases_df") is not None:
        return "Similar cases already retrieved for this query."
    df = compile_similar_cases(query)
    st.session_state["last_query"] = query
    st.session_state["similar_cases_df"] = df
    return f"Retrieved {len(df)} similar cases."


def generate_answer(question: str) -> str:
    """Generate an answer using grouped case and discussion content from cached similar cases."""
    df = st.session_state.get("similar_cases_df")
    if df is None or df.empty:
        return "No similar cases available. Please provide a clinical query first."
    context = "\n\n".join(
        f"article_id: {article_id}\n"
        f"cases: {' '.join(group['case_text'])}\n"
        f"discussion: {' '.join(group['discussion'])}"
        for article_id, group in df.groupby("article_id")
    )
    prompt = f"""
    You are a clinical assistant analyzing real-world clinical case reports.
    Below is a list of articles, each identified by a unique article_id. Each entry includes case descriptions and discussion content.
    {context}
    Using only the information above, answer the following question:
    {question}
    Your answer must be descriptive and clinically precise.
    Justify each clinical insight by citing the relevant article ID at the end of the sentence, using brackets like [PMC1234567].
    If the information above is insufficient to answer, tell the user that no relevant data was found, and invite the user to rephrase or to ask something else.
    Respond in English if the question is in English, or in Spanish if the question is in Spanish.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config= types.GenerateContentConfig(temperature=0.0)
    )
    return response


def format_text(response_text: str) -> str:
    """Format article ID references into numbered hyperlinks and append citation list, if references exist."""
    if not re.search(r"\[PMC\d+(?:,\s*PMC\d+)*\]", response_text):
        return response_text
    df = st.session_state.get("similar_cases_df")
    if df is None or df.empty:
        return response_text
    citations = df.groupby("article_id")["citation"].first().to_dict()
    link_map = {
        aid: f"https://pmc.ncbi.nlm.nih.gov/articles/{aid}/"
        for aid in citations.keys()}
    raw_refs = re.findall(r"\[(PMC\d+(?:,\s*PMC\d+)*)\]", response_text)
    ordered_ids = []
    for block in raw_refs:
        for aid in map(str.strip, block.split(",")):
            if aid not in ordered_ids:
                ordered_ids.append(aid)
    ref_map = {aid: idx + 1 for idx, aid in enumerate(ordered_ids)}
    def replace_refs(match):
        ids = [rid.strip() for rid in match.group(1).split(",")]
        return "[" + ", ".join(
            f"[{ref_map[aid]}]({link_map[aid]})" if link_map.get(aid) else str(ref_map[aid])
            for aid in ids
        ) + "]"
    formatted_text = re.sub(r"\[(PMC\d+(?:,\s*PMC\d+)*)\]", replace_refs, response_text)
    reference_list = "\n".join(
        f"{ref_map[aid]}. {citations.get(aid, 'Citation not found')}" for aid in ordered_ids
    )
    return f"{formatted_text}\n\n**References:**\n{reference_list}"


def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role


def export_cases() -> str:
    """Tool: export current similar cases and prepare a combined message with download button."""
    df = st.session_state.get("similar_cases_df")
    if df is None or df.empty:
        return "No similar cases available to export."
    export_df = df[["case_id", "case_text", "citation"]].copy()
    counter = st.session_state.get("csv_export_counter", 1)
    filename = f"similar_cases_{counter}.csv"
    st.session_state["csv_export_counter"] = counter + 1
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.session_state.chat_history.append({
        "role": "assistant",
        "type": "csv_download_message",
        "csv_data": csv_data,
        "filename": filename
    })
    return f"{len(export_df)} similar cases ready to export."


# Session state chat instance
if "chat_session" not in st.session_state:
    st.session_state.chat_session = client.chats.create(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            tools=[find_cases, generate_answer, format_text, export_cases],
            temperature=0.0,
            automatic_function_calling={"disable": False},
            system_instruction="""
            You are a clinical assistant specialized in retrieving and analyzing real-world medical case reports.
            - If the user provides a clinical case, retrieve similar cases using 'find_cases'.
            - If the user asks a question, answer it using 'generate_answer'.
            - If the user includes both a case and a question, run 'find_cases' and 'generate_answer' in sequence.
            - After 'generate_answer', always run 'format_text' to structure any article references found and to include the corresponding links.
            - Only run 'find_cases' again if the user clearly introduces a new case or changes the clinical context.
            - If the user asks a new, unrelated clinical question, re-run 'find_cases' before 'generate_answer' to ensure the answer is based on relevant evidence.
            - If the user asks to export, or implies interest in downloading results, call 'export_cases'.
            - Do not forget to add the corresponding links to the numbered hyperlinks.
            - Respond in English if the user question is in English, or in Spanish if the question is in Spanish.
            """
        )
    )

# Page components
with st.sidebar:
    st.logo("img/digital-twin-retriever-logo.webp", size="large")
    st.image("img/fingers-logo.webp")
    st.write("")
    st.write(
        """
        **Digital Twin Retriever** is an AI chatbot that helps users search and summarize clinical cases, 
        combining advanced retrieval and generative AI methods to support clinical decision-making. 
        It relies on [MultiCaRe dataset](https://zenodo.org/records/14994046), an open-access database of 
        over 90,000 de-identified reports from PubMed Central.
        """
    )
    st.write("")
    if st.button("🔄 New Chat"):
        st.session_state.chat_history = []
        st.session_state.chat_started = False
        st.rerun()
    st.write("")  
    st.write(
        """
        *Developed by [María Carolina González Galtier, MD, MA](https://www.linkedin.com/in/carogaltier/) &
        [Mauro Andrés Nievas Offidani, MD, MSc](https://www.linkedin.com/in/mauronievasoffidani/)*
        """
    )



user_prompt = st.chat_input("Enter a clinical case or ask a question:")

if user_prompt:
    st.session_state.chat_started = True
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    try:
        response = st.session_state.chat_session.send_message(user_prompt)
        response_text = getattr(response, "text", str(response))
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

    except Exception as e:
        error_msg = f"An error occurred: `{e}`"
        st.error(error_msg)
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message.get("type") == "csv_download_message" and message.get("csv_data") and message.get("filename"):
            st.download_button(
                label="📥 Download CSV",
                data=message["csv_data"],
                file_name=message["filename"],
                mime="text/csv",
                key=message["filename"]
            )
        else:
            st.markdown(message["content"])

if not st.session_state.chat_started:
    st.markdown(f"""
    <div class="custom-container">
        <img src='https://raw.githubusercontent.com/digital-twin-retriever/streamlit-app/refs/heads/main/img/robot.webp'>
        <h1>DIGITAL TWIN RETRIEVER</h1>
        <p>Start by asking a clinical question or providing a case description.</p>
        <p>Our AI will search for relevant cases and assist you with evidence-informed insights.</p>
    </div>
    """, unsafe_allow_html=True)
