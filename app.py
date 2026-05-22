import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from google import genai
from google.genai import types
from google.api_core import retry


# ---------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------

st.set_page_config(page_title="Digital Twin Retriever", page_icon=":robot_face:")

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

is_retriable = lambda e: (
    isinstance(e, genai.errors.APIError) and e.code in {429, 503}
)

if not hasattr(genai.models.Models.generate_content, "__wrapped__"):
    genai.models.Models.generate_content = retry.Retry(
        predicate=is_retriable
    )(genai.models.Models.generate_content)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

@st.cache_resource
def load_case_data() -> pd.DataFrame:
    return pd.read_parquet(
        "https://zenodo.org/records/20345273/files/case_texts.parquet?download=1"
    )


@st.cache_resource
def load_embedding_data() -> pd.DataFrame:
    return pd.read_parquet(
        "https://zenodo.org/records/20345273/files/case_embeddings.parquet?download=1"
    )


with st.spinner("Loading clinical case texts..."):
    case_df = load_case_data()

with st.spinner("Loading case embeddings..."):
    emb_df = load_embedding_data()


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

if "similar_cases_df" not in st.session_state:
    st.session_state.similar_cases_df = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

if "last_retrieval_query" not in st.session_state:
    st.session_state.last_retrieval_query = None

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ""

if "citation_cache" not in st.session_state:
    st.session_state.citation_cache = {}

if "csv_export_counter" not in st.session_state:
    st.session_state.csv_export_counter = 1


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def truncate_text(text: str, max_chars: int = 2500) -> str:
    """Safely truncate text to control prompt size."""

    if text is None:
        return ""

    text = str(text).strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def get_recent_turns(max_messages: int = 6, max_chars: int = 1800) -> str:
    """Return a compact recent conversation window."""

    messages = st.session_state.chat_history[-max_messages:]

    recent_text = "\n".join(
        f"{message.get('role', 'unknown')}: {message.get('content', '')}"
        for message in messages
        if message.get("content")
    )

    return truncate_text(recent_text, max_chars)


def build_contextual_retrieval_query(user_prompt: str) -> str:
    """
    Build one contextual retrieval query using:
    - compact conversation memory
    - recent turns
    - current user prompt

    This avoids fragile regex intent classification and keeps retrieval conversational.
    """

    memory = st.session_state.get("conversation_memory", "")
    recent_turns = get_recent_turns()

    query = f"""
Conversation memory:
{memory}

Recent conversation:
{recent_turns}

Current user question:
{user_prompt}
"""

    return truncate_text(query, max_chars=3000)


def update_conversation_memory(user_prompt: str, assistant_answer: str) -> None:
    """
    Update compact rolling memory without an extra LLM call.

    This keeps enough context for follow-up questions without storing the full chat forever.
    """

    previous_memory = st.session_state.get("conversation_memory", "")

    clean_answer = re.sub(r"\*\*References:\*\*.*", "", assistant_answer, flags=re.DOTALL)
    clean_answer = re.sub(r"\[[0-9,\s\]\(\)https:/.\-a-zA-Z]+\]", "", clean_answer)

    memory_update = f"""
{previous_memory}

Latest user question:
{truncate_text(user_prompt, 500)}

Latest assistant answer summary:
{truncate_text(clean_answer, 700)}
"""

    st.session_state.conversation_memory = truncate_text(memory_update, max_chars=1800)


def extract_pmc_ids(text: str) -> list[str]:
    """Extract unique PMC IDs from a text preserving order."""

    if not isinstance(text, str):
        text = str(text)

    ids = re.findall(r"PMC\d+", text)

    ordered_ids = []
    for pmcid in ids:
        if pmcid not in ordered_ids:
            ordered_ids.append(pmcid)

    return ordered_ids


# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------

def find_top_similar(
    query: str,
    top_k: int = 8,
    similarity_threshold: float = 0.60,
) -> pd.Series:
    """Return top_k most semantically similar cases above the threshold."""

    embedding_dim = emb_df.shape[1]

    # Kept inside the function to avoid Streamlit resource/cache issues.
    emb_values = emb_df.values.astype(np.float32)
    emb_norm = np.linalg.norm(emb_values, axis=1, keepdims=True)
    normed_embeddings = emb_values / np.maximum(emb_norm, 1e-12)

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=embedding_dim,
        ),
    )

    query_vector = np.array(response.embeddings[0].values, dtype=np.float32)
    query_vector = query_vector / max(np.linalg.norm(query_vector), 1e-12)

    scores = normed_embeddings @ query_vector
    result = pd.Series(scores, index=emb_df.index)

    return result[result >= similarity_threshold].nlargest(top_k)


def compile_similar_cases(
    retrieval_query: str,
    top_k: int = 8,
    similarity_threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Find similar clinical case chunks.

    This version does not fetch citations or discussion upfront.
    It keeps retrieval fast and only fetches citations later for PMCs actually cited.
    """

    top_similar = find_top_similar(
        retrieval_query,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    if top_similar.empty:
        # Fallback with a slightly lower threshold.
        top_similar = find_top_similar(
            retrieval_query,
            top_k=top_k,
            similarity_threshold=0.52,
        )

    if top_similar.empty:
        return pd.DataFrame()

    df = case_df.loc[case_df["case_id"].isin(top_similar.index)].copy()

    if df.empty:
        return pd.DataFrame()

    df["case_similarity_score"] = df["case_id"].map(top_similar)
    df.sort_values("case_similarity_score", ascending=False, inplace=True)

    expected_columns = ["case_id", "article_id", "case_text", "case_similarity_score"]
    available_columns = [col for col in expected_columns if col in df.columns]

    return df[available_columns].copy()


def retrieve_cases(user_prompt: str) -> pd.DataFrame:
    """Always run conversational retrieval using one contextual query."""

    retrieval_query = build_contextual_retrieval_query(user_prompt)

    if (
        st.session_state.last_retrieval_query == retrieval_query
        and st.session_state.similar_cases_df is not None
    ):
        return st.session_state.similar_cases_df

    df = compile_similar_cases(retrieval_query)

    st.session_state.last_retrieval_query = retrieval_query
    st.session_state.similar_cases_df = df

    return df


def build_retrieved_context(df: pd.DataFrame, max_case_chars: int = 2200) -> str:
    """Build compact retrieved context for the generation prompt."""

    if df is None or df.empty:
        return ""

    context_blocks = []

    for article_id, group in df.groupby("article_id", sort=False):
        similarity_score = group["case_similarity_score"].max()

        case_text = " ".join(
            group["case_text"].dropna().astype(str).tolist()
        )

        block = f"""
Article ID: {article_id}
Similarity score: {similarity_score:.3f}
Retrieved case text:
{truncate_text(case_text, max_case_chars)}
"""

        context_blocks.append(block.strip())

    return "\n\n---\n\n".join(context_blocks)


# ---------------------------------------------------------------------
# Citations and reference formatting
# ---------------------------------------------------------------------

def fetch_citation(pmcid: str) -> str:
    """Fetch APA-style citation for a given PMCID."""

    if pmcid in st.session_state.citation_cache:
        return st.session_state.citation_cache[pmcid]

    try:
        url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query=PMCID:{pmcid}&format=json"
        )

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        result = response.json()["resultList"]["result"][0]

        doi = result.get("doi")
        doi_text = f"https://doi.org/{doi}" if doi else ""

        citation = (
            f"{result.get('authorString', 'Unknown authors')} "
            f"({result.get('pubYear', 'n.d.')}). "
            f"{result.get('title', 'No title')} "
            f"*{result.get('journalTitle', 'Unknown journal')}*, "
            f"{result.get('journalVolume', '')}"
            f"({result.get('issue', '')}), "
            f"{result.get('pageInfo', '')}. "
            f"{doi_text}"
        ).strip()

    except Exception as e:
        citation = f"Citation not available for {pmcid}. Error: {e}"

    st.session_state.citation_cache[pmcid] = citation

    return citation


def format_text(response_text: str) -> str:
    """
    Convert PMC references into numbered hyperlinks and append reference list.

    Expected model citation format:
    [PMC1234567]
    [PMC1234567, PMC7654321]
    """

    if not isinstance(response_text, str):
        response_text = str(response_text)

    pattern = r"\[\s*(PMC\d+(?:\s*,\s*PMC\d+)*)\s*\]"

    if not re.search(pattern, response_text):
        return response_text

    raw_refs = re.findall(pattern, response_text)

    ordered_ids = []

    for block in raw_refs:
        for article_id in re.split(r"\s*,\s*", block.strip()):
            if article_id not in ordered_ids:
                ordered_ids.append(article_id)

    ref_map = {
        article_id: index + 1
        for index, article_id in enumerate(ordered_ids)
    }

    def replace_refs(match):
        ids = re.split(r"\s*,\s*", match.group(1).strip())

        formatted_refs = []

        for article_id in ids:
            number = ref_map.get(article_id)
            url = f"https://pmc.ncbi.nlm.nih.gov/articles/{article_id}/"

            if number:
                formatted_refs.append(f"[{number}]({url})")
            else:
                formatted_refs.append(article_id)

        return "[" + ", ".join(formatted_refs) + "]"

    formatted_text = re.sub(pattern, replace_refs, response_text)

    reference_list = "\n".join(
        f"{ref_map[article_id]}. {fetch_citation(article_id)}"
        for article_id in ordered_ids
    )

    return f"{formatted_text}\n\n**References:**\n\n{reference_list}"


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

def generate_answer(user_prompt: str) -> str:
    """Generate an answer using retrieved case chunks and conversation context."""

    df = st.session_state.get("similar_cases_df")

    if df is None or df.empty:
        return (
            "No relevant data was found in the retrieved cases. "
            "Please rephrase the clinical question or provide more clinical detail."
        )

    retrieved_context = build_retrieved_context(df)
    memory = st.session_state.get("conversation_memory", "")
    recent_turns = get_recent_turns()

    prompt = f"""
You are a clinical assistant analyzing real-world clinical case reports.

Conversation memory:
{memory}

Recent conversation:
{recent_turns}

Current user question:
{user_prompt}

Retrieved clinical case report chunks:
{retrieved_context}

Instructions:
- Answer in Spanish if the current user question is in Spanish.
- Answer in English if the current user question is in English.
- Use only the retrieved clinical case report chunks as evidence.
- First assess whether the retrieved chunks are clinically relevant to the current user question.
- If the retrieved chunks are unrelated or insufficient, say that no relevant data was found in the retrieved cases.
- Do not answer using unrelated retrieved chunks.
- Do not invent guideline-based recommendations if they are not supported by the retrieved chunks.
- Be clinically precise and clear.
- Every clinically relevant statement must cite at least one Article ID using this exact format: [PMC1234567].
- Only cite Article IDs that appear in the retrieved clinical case report chunks.
- Do not create a separate reference list. The application will add references automatically.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )

    return response.text


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def export_cases() -> None:
    """Prepare similar cases as downloadable CSV."""

    df = st.session_state.get("similar_cases_df")

    if df is None or df.empty:
        st.warning("No similar cases available to export.")
        return

    export_columns = [
        col for col in ["case_id", "article_id", "case_text", "case_similarity_score"]
        if col in df.columns
    ]

    export_df = df[export_columns].copy()

    counter = st.session_state.csv_export_counter
    filename = f"similar_cases_{counter}.csv"
    st.session_state.csv_export_counter += 1

    csv_data = export_df.to_csv(index=False).encode("utf-8")

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "type": "csv_download_message",
            "csv_data": csv_data,
            "filename": filename,
        }
    )


# ---------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

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

    if st.button("New Chat"):
        st.session_state.chat_history = []
        st.session_state.chat_started = False
        st.session_state.similar_cases_df = None
        st.session_state.last_retrieval_query = None
        st.session_state.conversation_memory = ""
        st.rerun()

    st.write("")

    st.write(
        """
        *Developed by [María Carolina González Galtier, MD, MA](https://www.linkedin.com/in/carogaltier/) &
        [Mauro Andrés Nievas Offidani, MD, MSc](https://www.linkedin.com/in/mauronievasoffidani/)*
        """
    )


# ---------------------------------------------------------------------
# Chat pipeline
# ---------------------------------------------------------------------

user_prompt = st.chat_input("Enter a clinical case or ask a question:")

if user_prompt:
    st.session_state.chat_started = True

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    try:
        with st.spinner("Retrieving relevant clinical cases..."):
            retrieve_cases(user_prompt)

        with st.spinner("Generating answer..."):
            raw_answer = generate_answer(user_prompt)
            final_answer = format_text(raw_answer)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": final_answer,
            }
        )

        update_conversation_memory(user_prompt, final_answer)

    except Exception as e:
        error_msg = f"An error occurred: `{e}`"
        st.error(error_msg)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": error_msg,
            }
        )


# ---------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if (
            message.get("type") == "csv_download_message"
            and message.get("csv_data")
            and message.get("filename")
        ):
            st.download_button(
                label="Download CSV",
                data=message["csv_data"],
                file_name=message["filename"],
                mime="text/csv",
                key=message["filename"],
            )
        else:
            st.markdown(message["content"])


# ---------------------------------------------------------------------
# Optional debug panel
# ---------------------------------------------------------------------

with st.expander("Retrieval debug", expanded=False):
    st.write("Last retrieval query:")
    st.code(st.session_state.get("last_retrieval_query", "") or "")

    df_debug = st.session_state.get("similar_cases_df")

    if df_debug is not None and not df_debug.empty:
        debug_columns = [
            col for col in ["case_id", "article_id", "case_similarity_score"]
            if col in df_debug.columns
        ]

        st.dataframe(df_debug[debug_columns])
    else:
        st.write("No retrieved cases yet.")


# ---------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------

if not st.session_state.chat_started:
    st.markdown(
        """
        <div class="custom-container">
            <img src='https://raw.githubusercontent.com/digital-twin-retriever/streamlit-app/refs/heads/main/img/robot.webp'>
            <h1>DIGITAL TWIN RETRIEVER</h1>
            <p>Start by asking a clinical question or providing a case description.</p>
            <p>Our AI will search for relevant cases and assist you with evidence-informed insights.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
