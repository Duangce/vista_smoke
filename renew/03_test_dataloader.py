"""
Standalone dataloader sanity check for the asphalt_wetness0 annotation.

This mimics Vista's BaseDataset + NuScenesDataset logic without requiring the
full vwm package, so we can verify the preprocessed data is loadable.
"""
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AsphaltDataset(Dataset):
    def __init__(self, data_root="/workspace/ws/Vista/renew",
                 anno_file="/workspace/ws/Vista/renew/annotations/asphalt_wetness0.json",
                 target_height=576, target_width=1024, num_frames=25):
        self.data_root = Path(data_root)

        assert target_height % 64 == 0 and target_width % 64 == 0, \
            "Resize to integer multiple of 64"
        self.img_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0)
        ])

        with open(anno_file, "r") as f:
            self.samples = json.load(f)

        self.target_height = target_height
        self.target_width = target_width
        self.num_frames = num_frames
        self.action_mod = 0

    def preprocess_image(self, image_path):
        image = Image.open(self.data_root / image_path)
        ori_w, ori_h = image.size
        if ori_w / ori_h > self.target_width / self.target_height:
            tmp_w = int(self.target_width / self.target_height * ori_h)
            left = (ori_w - tmp_w) // 2
            right = (ori_w + tmp_w) // 2
            image = image.crop((left, 0, right, ori_h))
        elif ori_w / ori_h < self.target_width / self.target_height:
            tmp_h = int(self.target_height / self.target_width * ori_w)
            top = (ori_h - tmp_h) // 2
            bottom = (ori_h + tmp_h) // 2
            image = image.crop((0, top, ori_w, bottom))
        image = image.resize((self.target_width, self.target_height), resample=Image.LANCZOS)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.img_preprocessor(image)
        return image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_dict = self.samples[index]
        self.action_mod = (self.action_mod + index) % 4

        image_seq = []
        for i in range(self.num_frames):
            img_path = sample_dict["frames"][i]
            image = self.preprocess_image(img_path)
            image_seq.append(image)

        cond_aug = torch.tensor([0.0])
        data_dict = {
            "img_seq": torch.stack(image_seq),
            "motion_bucket_id": torch.tensor([127]),
            "fps_id": torch.tensor([9]),
            "cond_frames_without_noise": image_seq[0],
            "cond_frames": image_seq[0] + cond_aug * torch.randn_like(image_seq[0]),
            "cond_aug": cond_aug
        }

        if self.action_mod == 0:
            data_dict["trajectory"] = torch.tensor(sample_dict["traj"][2:])
        elif self.action_mod == 1:
            data_dict["command"] = torch.tensor(sample_dict["cmd"])
        elif self.action_mod == 2:
            if sample_dict["speed"]:
                data_dict["speed"] = torch.tensor(sample_dict["speed"][1:])
            if sample_dict["angle"]:
                data_dict["angle"] = torch.tensor(sample_dict["angle"][1:]) / 780
        elif self.action_mod == 3:
            if sample_dict["z"] > 0:
                gx, gy = sample_dict["goal"]
                if 0 < gx < 1600 and 0 < gy < 900:
                    data_dict["goal"] = torch.tensor([gx / 1600, gy / 900])
        else:
            raise ValueError

        return data_dict


def main():
    ds = AsphaltDataset()
    print(f"Dataset size: {len(ds)}")

    for i in range(min(3, len(ds))):
        item = ds[i]
        print(f"\nSample {i}:")
        print(f"  img_seq shape: {item['img_seq'].shape}")
        print(f"  cond_frames shape: {item['cond_frames'].shape}")
        for key in ["trajectory", "command", "speed", "angle", "goal"]:
            if key in item:
                print(f"  {key}: {item[key].shape}  {item[key].tolist()}")


if __name__ == "__main__":
    main()
