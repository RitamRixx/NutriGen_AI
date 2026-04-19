import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document


def load_youtube_from_excel(excel_path: str):
    df = pd.read_excel(excel_path)
    docs  =[]
    for _, row in df.iterrows():
        video_id = row["link"].split("v=")[-1].split("&")[0]
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([t["text"] for t in transcript_list])
            doc = Document(
                page_content=transcript_text,
                metadata={
                    "title": row["Title"],
                    "source": row["link"],
                    "source_type": "youtube" 
                }
            )
            docs.append(doc)
        except Exception as e:
            print(f"Failed {row['Title']}: {e}")
    return docs
