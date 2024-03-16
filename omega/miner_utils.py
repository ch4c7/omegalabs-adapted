import os
import subprocess
import time
from typing import List, Tuple

from pytube import Search, YouTube
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

from omega.protocol import VideoMetadata
from omega.imagebind_wrapper import ImageBind
from omega.constants import MAX_VIDEO_LENGTH
from omega import video_utils

TWENTY_MINUTES = 1200

class VideoAnalyzer:
    def __init__(self, model_id, revision):
        self.model, self.tokenizer = self.initialize_moondream(model_id, revision)

    def initialize_moondream(self, model_id, revision):
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        return model, tokenizer

    def extract_frames(self, video_path, interval=1):
        output_folder = f"{video_path}_frames"
        os.makedirs(output_folder, exist_ok=True)
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps=1/{interval}',
            '-q:v', '2',
            f'{output_folder}/frame_%04d.jpg'
        ]
        subprocess.run(cmd)
        return output_folder

    def analyze_frame(self, frame_path):
        image = Image.open(frame_path)
        enc_image = self.model.encode_image(image)
        description = self.model.answer_question(enc_image, "Describe this image.", self.tokenizer)
        return description

    def aggregate_descriptions(self, descriptions):
        return " ".join(descriptions)

    def generate_video_description(self, video_path, interval=1):
        output_folder = self.extract_frames(video_path, interval)
        frame_paths = self.get_frame_paths(output_folder)
        descriptions = [self.analyze_frame(frame_path) for frame_path in frame_paths]
        return self.aggregate_descriptions(descriptions)

    def get_frame_paths(self, output_folder):
        frame_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.jpg')]
        return frame_files

def get_description(yt: YouTube, video_path: str) -> str:
    """
    Get / generate the description of a video. It first tries to generate a description
    using the Moondream model. If that fails, it falls back to the existing logic.
    """
    # Construct the original description based on title, description, and keywords
    original_description = yt.title
    if yt.description:
        original_description += f"\n\n{yt.description}"
    if yt.keywords:
        original_description += f"\n\nKeywords: {', '.join(yt.keywords)}"

    # Attempt to generate the enhanced description using the Moondream model
    try:
        video_analyzer = VideoAnalyzer("vikhyat/moondream2", "2024-03-06")
        enhanced_description = video_analyzer.generate_video_description(video_path)

        # Log both descriptions for comparison
        print(f"Original Description: {original_description}")
        print(f"Enhanced Description: {enhanced_description}")

        return enhanced_description
    except Exception as e:
        print(f"Failed to generate description using Moondream model: {e}")
        # If Moondream fails, log the original description and return it
        print(f"Returning Original Description: {original_description}")
        return original_description


def get_relevant_timestamps(query: str, yt: YouTube, video_path: str) -> Tuple[int, int]:
    """
    Get the optimal start and end timestamps (in seconds) of a video for ensuring relevance
    to the query.

    Miner TODO: Implement logic to get the optimal start and end timestamps of a video for
    ensuring relevance to the query.
    """
    start_time = 0
    end_time = min(yt.length, MAX_VIDEO_LENGTH)
    return start_time, end_time


def search_and_embed_videos(query: str, num_videos: int, imagebind: ImageBind) -> List[VideoMetadata]:
    """
    Search YouTube for videos matching the given query and return a list of VideoMetadata objects.

    Args:
        query (str): The query to search for.
        num_videos (int, optional): The number of videos to return.

    Returns:
        List[VideoMetadata]: A list of VideoMetadata objects representing the search results.
    """
    video_metas = []
    s = Search(query)
    try:
        while len(video_metas) < num_videos:
            for result in s.results:
                start = time.time()
                download_path = video_utils.download_video(
                    result.video_id,
                    start=0,
                    end=min(result.length, FIVE_MINUTES)  # download the first 5 minutes at most
                )
                if download_path:
                    print(f"Downloaded video {result.video_id} ({min(result.length, FIVE_MINUTES)}) in {time.time() - start} seconds")
                    clip_path = None
                    try:
                        start, end = get_relevant_timestamps(query, result, download_path)
                        description = get_description(result, download_path)
                        print(f"{description}")
                        clip_path = video_utils.clip_video(download_path.name, start, end)
                        embeddings = imagebind.embed([description], [clip_path])
                        video_metas.append(VideoMetadata(
                            video_id=result.video_id,
                            description=description,
                            views=result.views,
                            start_time=start,
                            end_time=end,
                            video_emb=embeddings.video[0].tolist(),
                            audio_emb=embeddings.audio[0].tolist(),
                            description_emb=embeddings.description[0].tolist(),
                        ))
                    finally:
                        download_path.close()
                        if clip_path:
                            clip_path.close()
                if len(video_metas) == num_videos:
                    break
            s.get_next_results()

    except Exception as e:
        print(f"Error searching for videos: {e}")
    
    return video_metas
