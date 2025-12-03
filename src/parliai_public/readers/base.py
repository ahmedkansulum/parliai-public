"""Base class for other readers to inherit from."""

import abc
import datetime as dt
import json
import os
import re
from importlib import resources
from typing import Iterable
from urllib.parse import urlparse

import requests
import toml
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama

from parliai_public import dates


class BaseReader(metaclass=abc.ABCMeta):
    """
    A base class for readers to inherit.

    This class is not to be used in practice except for inheritance.

    To make your own reader class, you can inherit from this base class
    and implement the following methods:

    - `retrieve_latest_entries`
    - `_read_metadata`
    - `_read_contents`
    - `render`

    Parameters
    ----------
    urls : list[str]
        List of URLs.
    terms : Iterable[str], optional
        Key terms to filter content on.
    dates : list[dt.date], optional
        Dates to pull entries from.
    outdir : str
        Output directory.
    prompt : str, optional
        LLM prompt.
    llm_name : str, optional
        Name of the model in Ollama.
    llm : ChatOllama, optional
        Model wrapper.
    """

    _default_config: str = "base.toml"
    _source: None | str = None

    def __init__(
        self,
        urls: list[str],
        terms: None | Iterable[str] = None,
        dates: None | list[dt.date] = None,
        outdir: str = "out",
        prompt: None | str = None,
        llm_name: None | str = None,
        llm: None | ChatOllama = None,
        inconsistency_statement: None | str = None,
    ) -> None:
        """Initialise BaseReader."""
        self.urls = urls
        base_config = toml.load("src/parliai_public/_config/base.toml")
        self.terms = terms or base_config["keywords"]
        self.inconsistency_statement = (
            inconsistency_statement or base_config["inconsistency_statement"]
        )
        self.dates = dates or [dt.date.today() - dt.timedelta(days=1)]
        self.outdir = outdir

        config = self._load_config()
        self.prompt = prompt or config["prompt"]
        self.llm_name = llm_name or config["llm_name"]
        self.llm = llm

    @classmethod
    def _load_config(cls, path: None | str = None) -> dict:
        """Load a configuration file."""
        if isinstance(path, str):
            return toml.load(path)

        where = resources.files("parliai_public._config")
        with resources.as_file(where.joinpath(cls._default_config)) as c:
            config = toml.load(c)

        return config

    @classmethod
    def from_toml(cls, path: None | str = None) -> "BaseReader":
        """Create an instance using a TOML file."""
        config = cls._load_config(path)

        start = config.pop("start", None)
        end = config.pop("end", None)
        window = config.pop("window", None)
        form = config.pop("form", "%Y-%m-%d")

        config["dates"] = None
        if start or end or window:
            config["dates"] = dates.list_dates(start, end, window, form)

        return cls(**config)

    def check_contains_terms(self, string: str) -> bool:
        """Check whether a string contains any search terms."""
        terms = self.terms
        if not terms:
            return True

        string = string.lower()
        for term in map(str.lower, terms):
            match = re.search(
                rf"(^|(?<=[\('\[\s])){term}(?=[\)\]\s!?.,:;'-]|$)", string
            )
            if match:
                return True

        return False

    def make_outdir(self) -> None:
        """Create the output directory for a run."""
        start, end = min(self.dates), max(self.dates)
        period = ".".join(map(dt.date.isoformat, [start, end]))
        name = ".".join((period, self.llm_name))

        outdir = os.path.join(self.outdir, name)
        outdir = self._tag_outdir(outdir)

        os.makedirs(outdir)
        self.outdir = outdir

    def _tag_outdir(self, outdir: str) -> str:
        """Add an incremental tag if output directory already exists."""
        if not os.path.exists(outdir):
            return outdir

        tag = 1
        while os.path.exists(updated := ".".join((outdir, str(tag)))):
            tag += 1

        return updated

    @abc.abstractmethod
    def retrieve_latest_entries(self) -> list[str]:
        """Retrieve URLs of the latest pages."""

    def get(self, url: str, check: bool = True) -> None | BeautifulSoup:
        """Retrieve HTML soup for a webpage."""
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        if (not check) or (
            check and self.check_contains_terms(soup.get_text())
        ):
            return soup
        return None

    def read(self, url: str) -> None | dict:
        """Read metadata + contents if page matches search criteria."""
        soup = self.get(url)
        page = None
        if soup is not None:
            metadata = self._read_metadata(url, soup)
            contents = self._read_contents(soup)
            page = {**metadata, **contents}
        return page

    @abc.abstractmethod
    def _read_metadata(self, url: str, soup: BeautifulSoup) -> dict:
        """Extract metadata from HTML."""

    @abc.abstractmethod
    def _read_contents(self, soup: BeautifulSoup) -> dict:
        """Extract text from HTML."""

    def instantiate_llm(self) -> None:
        """Instantiate LLM object."""
        self.llm_name = "gemma"
        self.llm = ChatOllama(model=self.llm_name, temperature=0)
        return None

    def analyse(self, transcript: dict) -> dict:
        """Send text chunks to LLM for analysis."""
        chunks = self._split_text_into_chunks(transcript["text"])

        responses = []
        for chunk in chunks:
            if self.check_contains_terms(chunk.page_content):
                response = self._analyse_chunk(chunk)

                if not self._check_response(response, chunk):
                    response += f"\n\n{self.inconsistency_statement}"

                responses.append(response)

        transcript["response"] = "\n\n".join(responses)
        return transcript

    def clean_response(self, response: str):
        """Remove unwanted prefix in Gemma model responses."""
        response = re.sub(r"^Sure(.*?:)\s*", "", response)
        return response

    @staticmethod
    def _split_text_into_chunks(
        text: str,
        sep: str = ". ",
        size: int = 4000,
        overlap: int = 1000,
    ) -> list[Document]:
        """Split long text into LLM-friendly chunks."""
        splitter = RecursiveCharacterTextSplitter(
            separators=sep,
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            keep_separator=False,
            is_separator_regex=False,
        )
        return splitter.create_documents([text])

    def _analyse_chunk(self, chunk: Document) -> str:
        """Analyse one chunk using an LLM."""
        prompt_template = PromptTemplate(
            input_variables=["keywords", "text"],
            template=self.prompt,
        )
        prompt = prompt_template.format(
            keywords=self.terms, text=chunk.page_content
        )

        llm = self.llm
        response = llm.invoke(prompt).content.strip()

        if self.llm_name == "gemma":
            response = self.clean_response(response)

        return response

    @staticmethod
    def _normalise_text(text: str) -> str:
        """Normalise text for comparison."""
        text = text.lower()
        return re.sub(r"[^\w\s]", "", text)

    def _check_response(self, response: str, chunk: Document) -> bool:
        """
        Check if LLM response appears verbatim in original text.

        Parameters
        ----------
        response : str
            LLM response.
        chunk : Document
            Original text chunk.

        Returns
        -------
        passed : bool
            True if LLM response text appears in original.
        """
        original = self._normalise_text(chunk.page_content)

        for el in response.split(". "):
            normalised_el = self._normalise_text(el)
            if not normalised_el:
                continue
            if normalised_el not in original:
                return False

        return True

    def save(self, page: dict) -> None:
        """
        Save an HTML entry in compact JSON format.

        The file is stored inside an output directory based on category
        and index.
        """
        cat, idx = page.get("cat"), page.get("idx")

        root = os.path.join(self.outdir, "data")
        where = root if cat is None else os.path.join(root, cat)
        os.makedirs(where, exist_ok=True)

        with open(os.path.join(where, f"{idx}.json"), "w") as f:
            json.dump(page, f, indent=4)

    @abc.abstractmethod
    def render(self, transcript: dict) -> str:
        """Render a Markdown summary of an entry."""

    def make_header(self, urls: list[str] = None) -> str:
        """Make header for summary output."""
        form = "%a, %d %b %Y"
        today = dt.date.today().strftime(form)

        dates = self.dates
        if len(dates) == 1:
            period = dates[-1].strftime(form)
        else:
            start = min(dates).strftime(form)
            end = max(dates).strftime(form)
            period = f"{start} to {end}"

        urls = urls or self.urls
        source = f"Based on information from {self._source}:\n"

        links = []
        for url in urls:
            parsed = urlparse(url)
            link = url.replace(f"{parsed.scheme}://", "", 1)
            links.append(f"- [{link}]({url})")

        header = "\n".join(
            (
                f"Publication date: {today}",
                f"Period covered: {period}",
                f"Search terms: {self.terms}",
                f"Model used: {self.llm_name}",
                "\n".join((source, *links)),
            )
        )

        return header
