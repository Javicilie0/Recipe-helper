import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl
from logger import (Colors, log_error, log_header, log_info, log_success)
load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorsstore = PineconeVectorStore(embedding = embeddings,index_name=os.environ['INDEX_NAME'])

tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    batches = [
         documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    async def add_batch(batch: List[Document], batch_num:int):
        try:
            await vectorsstore.aadd_documents(batch)
            log_success(
                    f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
                )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks)

    successful = sum(1 for r in results if r is True)
    log_success(f"Done: {successful}/{len(batches)} batches successful")

async def crawl_url(url: str) -> List[Document]:
    
    tavily_crawl_results = tavily_crawl.invoke(
        input= {
            "url": url,
            "extract_depth": "advanced",
            "instructions": "Recipes with ingredients and instructions",
            "max_depth": 1,
        }
    )
    if tavily_crawl_results.get("error"):
        log_error(f"TavilyCrawl: {tavily_crawl_results['error']}")
        return []
    else:
        log_success(f"TavilyCrawl: Successfully crawled {len(tavily_crawl_results['results'])}")
    all_docs = []

    for tavily_crawl_item in tavily_crawl_results["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_item['url']} from recipe site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_item["raw_content"],
                metadata ={"source": tavily_crawl_item["url"]}
            )
        )
    return all_docs

async def main():
    """Main async function to orchestrate the entire process."""
  
    log_header("Starting Recipe Ingestion Process")

    urls = [
        "https://www.simplyrecipes.com/recipes/",
        "https://www.bbcgoodfood.com/recipes/",
    ]

    tasks = [crawl_url(url) for url in urls]
    results = await asyncio.gather(*tasks)

    all_docs = []

    for doc_list in results:
        all_docs.extend(doc_list)

    log_info(f"Total documents collected: {len(all_docs)}")

    log_header("Splitting Documents into Chunks")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Successfully Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    await index_documents_async(splitted_docs, batch_size=500)
    log_success("Ingestion complete!")
   
if __name__ == "__main__":
    asyncio.run(main())