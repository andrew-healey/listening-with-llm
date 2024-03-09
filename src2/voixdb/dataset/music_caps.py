# TODO: generate musicsynth dataset ONLY and train on that
import csv
import multiprocessing
import pathlib as pl
import subprocess
from tqdm import tqdm

import torch
import os

from .. import util

ROOT = pl.Path("")

CSV = ROOT / "musiccaps-public.csv"
CSV_FILTERED = ROOT / "filtered.csv"
AUDIO_DIR = ROOT / "videos"

class MusicCapsDataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, entries, tokenizer, prompt_template, audio_encoder,embed_tokens,max_length=1000,key=None):
        self.entries = entries
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_length = max_length

        self.audio_encoder = audio_encoder
        self.embed_tokens = embed_tokens
    
        filename = f"{key}.pt"
        if key is not None and os.path.exists(filename):
            self.raw_audio_embeds = torch.load(filename)
        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda",dtype=torch.float16):
                    print("Preloading raw audio embeddings:")
                    self.raw_audio_embeds = []
                    for i in tqdm(range(0, len(entries), 32)):
                        batch = entries[i:i+32]
                        batch_embeds = audio_encoder(torch.cat([util.load_audio_mels(entry["file"]) for entry in batch]).cuda()).cpu()
                        self.raw_audio_embeds.extend(batch_embeds)
            
            if key is not None:
                torch.save(self.raw_audio_embeds,filename)
        
        self.raw_sizes = set()

    def __len__(self):
        return len(self.entries)

    @torch.no_grad()
    def __getitem__(self, index):
        "Generates one sample of data"
        entry = self.entries[index]

        # mels = util.load_audio_mels(entry["file"])
        audio_embeds = self.raw_audio_embeds[index]
        caption = entry["caption"]

        prompt_ids, prompt_attention_mask = util.text_2_ids_and_attention_mask(
            self.tokenizer,
            self.prompt_template(),
        )
        end_prompt_ids, end_prompt_attention_mask = util.text_2_ids_and_attention_mask(
            self.tokenizer,
            util.end_template(),
            truncate=True,
        )
        cap_ids, cap_attention_mask = util.text_2_ids_and_attention_mask(
            self.tokenizer,
            caption,
            truncate=True,
        )

        # print(prompt_attention_mask.shape)

        assert len(prompt_attention_mask.shape) == 2

        attention_mask = torch.concat(
            (
                prompt_attention_mask,
                torch.ones(1,len(audio_embeds)//4),
                end_prompt_attention_mask,
                cap_attention_mask,
            ),
            dim=1
        )


        cap_embeds = self.embed_tokens(cap_ids)
        prompt_embeds = self.embed_tokens(prompt_ids)
        end_prompt_embeds = self.embed_tokens(end_prompt_ids)

        audio_embeds_standin = torch.zeros(len(audio_embeds),prompt_embeds.shape[1])

        language_C = prompt_embeds.shape[-1]
        audio_C = audio_embeds.shape[-1]

        # print(prompt_embeds.shape, torch.zeros(1,len(audio_embeds),language_C).shape,end_prompt_embeds.shape)

        inputs_embeds_raw = torch.concat(
            (
                prompt_embeds,
                torch.zeros(1,len(audio_embeds) // 4,language_C),
                end_prompt_embeds,
                cap_embeds,
            ),
            dim=1
        )

        inputs_raw = torch.concat(
            (
                prompt_ids,
                torch.zeros(1,len(audio_embeds) // 4),
                end_prompt_ids,
                cap_ids,
            ),
            dim=1
        )

        # print(prompt_embeds.shape,audio_embeds.shape)

        audio_embeds_raw = torch.concat(
            (
                torch.zeros(prompt_embeds.shape[1] * 4,audio_C),
                audio_embeds,
                torch.zeros(end_prompt_embeds.shape[1] * 4,audio_C),
                torch.zeros(cap_embeds.shape[1] * 4,audio_C)
            ),
            dim=0
        )

        # print("inputs_embeds_raw",
        #     (
        #         prompt_embeds.shape,
        #         torch.zeros(1,len(audio_embeds) // 4,language_C).shape,
        #         end_prompt_embeds.shape,
        #         cap_embeds.shape,
        #     ),
        #       )

        # print("audio_embeds_raw",
        #     (
        #         torch.zeros(prompt_embeds.shape[1] * 4,audio_C).shape,
        #         audio_embeds.shape,
        #         torch.zeros(end_prompt_embeds.shape[1] * 4,audio_C).shape,
        #         torch.zeros(cap_embeds.shape[1] * 4,audio_C).shape
        #     ),
        #     )

        audio_start = len(prompt_embeds)
        audio_end = len(prompt_embeds) + len(audio_embeds) // 4
        raw_length = inputs_embeds_raw.shape[1]
        cap_start = raw_length - cap_embeds.shape[1]

        # pad/truncate both

        pad = self.max_length - raw_length

        audio_start = max(0,audio_start + pad)
        audio_end = max(0,audio_end + pad)
        cap_start = max(0,cap_start + pad)

        self.raw_sizes.add(len(inputs_embeds_raw))

        # print("pre-pad",inputs_embeds_raw.shape,audio_embeds_raw.shape)

        if raw_length < self.max_length:
            inputs_embeds_raw = torch.concat(
                (torch.zeros(1,pad,inputs_embeds_raw.shape[-1]),inputs_embeds_raw),dim=1
            )
            audio_embeds_raw = torch.concat(
                (torch.zeros(pad*4,audio_embeds_raw.shape[1]),audio_embeds_raw)
            )
            attention_mask = torch.concat(
                (torch.zeros(1,pad),attention_mask),dim=1
            )
            inputs_raw = torch.concat(
                (torch.zeros(1,pad),inputs_raw),dim=1
            )

            audio_start += pad
            audio_end += pad

            # print("post-pad",inputs_embeds_raw.shape,audio_embeds_raw.shape)
        elif raw_length > self.max_length:
            inputs_embeds_raw = inputs_embeds_raw[:,-self.max_length:]
            inputs_raw = inputs_raw[:,-self.max_length:]
            audio_embeds_raw = audio_embeds_raw[-self.max_length*4:]
            attention_mask = attention_mask[:,-self.max_length:]
            # print("post-truncate",inputs_embeds_raw.shape,audio_embeds_raw.shape)

        ret = {
            "attention_mask": attention_mask.squeeze(0),
            "inputs_embeds_raw": inputs_embeds_raw.squeeze(0),
            "inputs_raw": inputs_raw.squeeze(0),
            "audio_embeds_raw": audio_embeds_raw,
            "audio_start":torch.tensor(audio_start),
            "audio_end":torch.tensor(audio_end),
            "cap_start":torch.tensor(cap_start),
        }

        # print("ret",{k:v.shape for k,v in ret.items()})
        # raise 1
        return ret


def load_csv(load_raw=False):
    if load_raw:
        entries = []
        mp3s = []
        pool = multiprocessing.Pool(6)

        with open(CSV, mode="r") as csv_file:
            # Create a CSV reader
            csv_reader = csv.DictReader(csv_file)
            # Iterate over each row in the CSV file
            for i, row in enumerate(csv_reader):
                # Each row is a dictionary where the keys are the column names
                file_path = AUDIO_DIR / f"{row['ytid']}-{i}.mp3"
                row["file"] = file_path
                entries.append(row)
                mp3s.append(file_path)

        res = pool.map(verify_mp3, mp3s)
        filtered = []
        for i, is_valid in enumerate(res):
            if is_valid:
                filtered.append(entries[i])

        # save filtered
        # Writing to CSV file
        with open(CSV_FILTERED, mode="w", newline="") as file:
            field_names = list(filtered[0].keys())
            writer = csv.DictWriter(file, fieldnames=field_names)
            # Write header
            writer.writeheader()
            # Write data
            writer.writerows(filtered)

        return filtered
    else:
        with open(CSV_FILTERED, mode="r") as csv_file:
            # Create a CSV reader
            csv_reader = csv.DictReader(csv_file)
            return list(csv_reader)


def verify_mp3(output_path_wt_suffix) -> bool:
    command = ["ffmpeg", "-v", "error", "-i", output_path_wt_suffix, "-f", "null", "-"]
    result = subprocess.run(command, capture_output=True, text=True)
    # Check the return code to see if there were any issues
    return result.returncode == 0
