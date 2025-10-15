import os
from argparse import ArgumentParser
import json

import zarr
import h5py
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument(
    "-i", "--input_path", type=str,
    help="The filepath to the PushT dataset"
)
parser.add_argument(
    "-o", "--output_path", type=str,
    help="The filepath for the output hdf5 dataset"
)


def convert_pusht_zarr_to_hdf5(pusht_dataset: zarr.Group, output_filepath):
    episode_ends = pusht_dataset.meta.episode_ends
    total_actions = pusht_dataset.data.action
    total_imgs = pusht_dataset.data.img
    total_keypoints = pusht_dataset.data.keypoint
    total_n_contacts = pusht_dataset.data.n_contacts
    total_states = pusht_dataset.data.state
    start = 0
    with h5py.File(output_filepath, "w") as f:
        data_group = f.create_group("data")

        for demo_no, end in tqdm(
            enumerate(episode_ends),
            total=len(episode_ends),
            desc="Converting zarr into hdf5"
        ):
            action = total_actions[start:end]
            img = total_imgs[start:end]
            keypoint = total_keypoints[start:end]
            n_contacts = total_n_contacts[start:end]
            state = total_states[start:end]
            demo_group = data_group.create_group(f"demo_{demo_no}")
            obs_group = demo_group.create_group("obs")
            demo_group.create_dataset("actions", data=action)
            obs_group.create_dataset("image", data=img)
            obs_group.create_dataset("n_contacts", data=n_contacts)
            obs_group.create_dataset("keypoints", data=keypoint)
            obs_group.create_dataset("state", data=state)

            demo_group.attrs["num_samples"] = end - start

            start = end

        env_args = {
            "env_name": "PushT",
            "type": 4,
            "render": False,
            "camera_heights": 96,
            "camera_widths": 96,
            "camera_names": ["birdseye"],
            "env_kwargs": {
                "damping": 0.0,
                "legacy": False,
                "render_size": 96,
                "reset_to_state": False
            },
            "render_gpu_device_id": 0,
            "camera_depths": False,
            "reward_shaping": False
        }
        data_group.attrs["env_args"] = json.dumps(env_args)

    print(f"Stored hdf5 dataset at {output_filepath}")


if __name__ == "__main__":
    args = parser.parse_args()
    input = args.input_path
    assert os.path.exists(input), f"PushT filepath {input} does not exist."
    pusht_dataset = zarr.open(input, "r")
    output = args.output_path
    convert_pusht_zarr_to_hdf5(pusht_dataset, output)
