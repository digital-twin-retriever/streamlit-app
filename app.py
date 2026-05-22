import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
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

if "last_query" not in st.session_state:
    st.session_state.last_query = None

if "csv_export_counter" not in st.session_state:
    st.session_state.csv_export_counter = 1


# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------

def find_top_similar(
    query: str,
    top_k: int = 10,
    similarity_threshold: float = 0.68,
) -> pd.Series:
    """Return top_k most semantically similar cases above the threshold."""

    embedding_dim = emb_df.shape[1]

    # Keep this inside the function to avoid Streamlit cache/resource issues.
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


def fetch_citation(pmcid: str) -> str:
    """Fetch APA-style citation for a given PMCID."""

    try:
        url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query=PMCID:{pmcid}&format=json"
        )
        result = requests.get(url, timeout=10).json()["resultList"]["result"][0]

        doi = result.get("doi")
        doi_text = f"https://doi.org/{doi}" if doi else ""

        return (
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
        return f"Citation not available for {pmcid}. Error: {e}"


def fetch_discussion(pmcid: str) -> str:
    """Return discussion section from a PMC article."""

    url = (
        "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/"
        f"pmcoa.cgi/BioC_xml/{pmcid}/ascii"
    )

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return f"Discussion not available for {pmcid}."

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
        discussion = []

        for i, p in enumerate(passages):
            if (
                p.find("infon", {"key": "type"}, string=re.compile("title", re.I))
                and p.find("text", string=re.compile("discuss", re.I))
            ):
                discussion = [
                    next_p.find("text").get_text(strip=True)
                    for next_p in passages[i + 1:]
                    if (
                        next_p.find("infon", {"key": "type"}, string="paragraph")
                        and next_p.find("text")
                    )
                ]
                break

        return " ".join(discussion) if discussion else "Discussion not found."

    except Exception as e:
        return f"Error processing discussion for {pmcid}: {e}"


def get_case_data(pmcids: list[str], max_workers: int = 5) -> dict:
    """Parallel fetch citation and discussion info for PMCIDs."""

    def fetch(pmcid: str):
        return pmcid, fetch_citation(pmcid), fetch_discussion(pmcid)

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch, pmcid) for pmcid in pmcids]

        for future in as_completed(futures):
            try:
                pmcid, citation, discussion = future.result()
                results[pmcid] = {
                    "citation": citation,
                    "discussion": discussion,
                }
            except Exception as e:
                results[pmcid] = {
                    "citation": f"Citation error: {e}",
                    "discussion": "",
                }

    return results


def compile_similar_cases(input_query: str) -> pd.DataFrame:
    """Find and enrich similar clinical cases with citations and discussions."""

    top_similar = find_top_similar(input_query)

    if top_similar.empty:
        return pd.DataFrame()

    df = case_df.loc[case_df["case_id"].isin(top_similar.index)].copy()
    df["case_similarity_score"] = df["case_id"].map(top_similar)
    df.sort_values("case_similarity_score", ascending=False, inplace=True)

    pmcids = (
        pd.Series(top_similar.index)
        .str.extract(r"(PMC\d+)")[0]
        .dropna()
        .unique()
        .tolist()
    )

    article_info = get_case_data(pmcids)

    df["citation"] = df["article_id"].map(
        lambda x: article_info.get(x, {}).get("citation", "Citation not available.")
    )

    df["discussion"] = df["article_id"].map(
        lambda x: article_info.get(x, {}).get("discussion", "Discussion not available.")
    )

    return df


def find_cases(query: str) -> pd.DataFrame:
    """Retrieve and store similar cases for a query."""

    if (
        st.session_state.last_query == query
        and st.session_state.similar_cases_df is not None
    ):
        return st.session_state.similar_cases_df

    df = compile_similar_cases(query)

    st.session_state.last_query = query
    st.session_state.similar_cases_df = df

    return df


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

def generate_answer(question: str) -> str:
    """Generate an answer using cached similar cases."""

    df = st.session_state.get("similar_cases_df")

    if df is None or df.empty:
        return (
            "No similar clinical cases were found for this query. "
            "Please rephrase the case or provide more clinical detail."
        )

    context = "\n\n".join(
        f"article_id: {article_id}\n"
        f"cases: {' '.join(group['case_text'].dropna().astype(str))}\n"
        f"discussion: {' '.join(group['discussion'].dropna().astype(str))}"
        for article_id, group in df.groupby("article_id")
    )

    prompt = f"""
You are a clinical assistant analyzing real-world clinical case reports.

Use only the information provided in the clinical case report context below.

Context:
{context}

Question:
{question}

Instructions:
- Answer in Spanish if the question is in Spanish.
- Answer in English if the question is in English.
- Be clinically precise.
- Do not invent recommendations that are not supported by the context.
- Cite every clinically relevant statement using the article ID in brackets, for example [PMC1234567].
- If the context is insufficient, say that no relevant data was found in the retrieved cases.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )

    return response.text


# ---------------------------------------------------------------------
# Reference formatting
# ---------------------------------------------------------------------

def format_text(response_text: str) -> str:
    """Convert PMC references into numbered hyperlinks and append reference list."""

    if not isinstance(response_text, str):
        response_text = str(response_text)

    pattern = r"\[\s*(PMC\d+(?:\s*,\s*PMC\d+)*)\s*\]"

    if not re.search(pattern, response_text):
        return response_text

    df = st.session_state.get("similar_cases_df")

    if df is None or df.empty:
        return response_text

    citations = df.groupby("article_id")["citation"].first().to_dict()

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
        f"{ref_map[article_id]}. {citations.get(article_id, 'Citation not found')}"
        for article_id in ordered_ids
    )

    return f"{formatted_text}\n\n**References:**\n\n{reference_list}"


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def export_cases() -> None:
    """Prepare similar cases as downloadable CSV."""

    df = st.session_state.get("similar_cases_df")

    if df is None or df.empty:
        st.warning("No similar cases available to export.")
        return

    export_df = df[["case_id", "case_text", "citation"]].copy()

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
        st.session_state.last_query = None
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
        with st.spinner("Retrieving similar clinical cases..."):
            find_cases(user_prompt)

        with st.spinner("Generating answer..."):
            raw_answer = generate_answer(user_prompt)
            final_answer = format_text(raw_answer)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": final_answer,
            }
        )

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
            content = message["content"]

            if message["role"] == "assistant":
                content = format_text(content)

            st.markdown(content)


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
