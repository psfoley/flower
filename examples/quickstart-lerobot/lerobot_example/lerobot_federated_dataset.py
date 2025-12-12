"""lerobot_example: A Flower / Hugging Face LeRobot app."""

from pathlib import Path
from typing import Callable

from datasets import Dataset
from lerobot.datasets.lerobot_dataset import (
    CODEBASE_VERSION,
    LeRobotDataset,
)
from lerobot.datasets.utils import (
    DATA_DIR,
    load_info,
    load_stats,
)
from lerobot.datasets.push_dataset_to_hub.utils import calculate_episode_data_index


class FilteredLeRobotDataset(LeRobotDataset):
    """Behaves like `LeRobotDataset` but using the dataset partition passed during
    construction."""

    def __init__(
        self,
        repo_id: str,
        root: Path | None = DATA_DIR,
        split: str = "train",
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        video_backend: str | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root = root,
            episodes = episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend)
        self.split = split

        # after filtering, the stored episode data index may not be the same
        # so let's calculate it on the filtered data
        #self.hf_dataset = reset_episode_index(self.hf_dataset)

        self.stats = load_stats(Path(self.root))
        #self.info = load_info(Path(self.root))
        #if self.video:
        #    self.videos_dir = load_videos(self.repo_id, CODEBASE_VERSION, self.root)
        #    self.video_backend = (
        #        self.video_backend if self.video_backend is not None else "pyav"
        #    )
