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
        self.entries = sorted(entries,key=lambda entry: entry['file'])
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
        self.raw_audio_sizes = set()

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

        n_audio_text_embeds = 250

        assert len(prompt_attention_mask.shape) == 2

        attention_mask = torch.concat(
            (
                prompt_attention_mask,
                torch.ones(1,n_audio_text_embeds),
                end_prompt_attention_mask,
                cap_attention_mask,
            ),
            dim=1
        )


        cap_embeds = self.embed_tokens(cap_ids)
        prompt_embeds = self.embed_tokens(prompt_ids)
        end_prompt_embeds = self.embed_tokens(end_prompt_ids)

        language_C = prompt_embeds.shape[-1]
        audio_C = audio_embeds.shape[-1]

        inputs_embeds_raw = torch.concat(
            (
                prompt_embeds,
                torch.zeros(1,n_audio_text_embeds,language_C),
                end_prompt_embeds,
                cap_embeds,
            ),
            dim=1
        )

        input_ids_raw = torch.concat(
            (
                prompt_ids,
                torch.zeros(1,n_audio_text_embeds),
                end_prompt_ids,
                cap_ids,
            ),
            dim=1
        )

        # treated totally separately
        n_audio_embeds = len(audio_embeds)
        audio_max = 2000 # TODO check if this is enough

        audio_embeds_raw = torch.concat(
            (
                audio_embeds,
            ),
            dim=0
        )

        audio_tokens_start = prompt_embeds.shape[1]
        audio_tokens_end = prompt_embeds.shape[1] + n_audio_text_embeds
        raw_token_length = inputs_embeds_raw.shape[1]
        cap_tokens_start = raw_token_length - cap_embeds.shape[1]

        # pad/truncate both
        token_pad = max(0,self.max_length - raw_token_length)

        audio_tokens_start = min(self.max_length,max(0,audio_tokens_start + token_pad))
        audio_tokens_end = min(self.max_length,max(0,audio_tokens_end + token_pad))
        assert audio_tokens_end == audio_tokens_start + n_audio_text_embeds, f"Padding bit into the fixed-size audio embed. This is not supported. raw_token_length = {raw_token_length}, audio_tokens_start = {audio_tokens_start}, audio_tokens_end = {audio_tokens_end}. text = '{self.prompt_template()}'+<audio tokens>"

        cap_tokens_start = max(0,cap_tokens_start + token_pad)

        self.raw_sizes.add(inputs_embeds_raw.shape[1])
        self.raw_audio_sizes.add(n_audio_embeds)

        # pad/truncate LLM inputs
        # pad on the left
        if raw_token_length < self.max_length:
            inputs_embeds_raw = torch.concat(
                (torch.zeros(1,token_pad,inputs_embeds_raw.shape[-1]),inputs_embeds_raw),dim=1
            )
            attention_mask = torch.concat(
                (torch.zeros(1,token_pad),attention_mask),dim=1
            )
            input_ids_raw = torch.concat(
                (torch.zeros(1,token_pad),input_ids_raw),dim=1
            )
        # overflow on the right
        elif raw_token_length > self.max_length:
            inputs_embeds_raw = inputs_embeds_raw[:,:self.max_length]
            input_ids_raw = input_ids_raw[:,:self.max_length]
            attention_mask = attention_mask[:,:self.max_length]
        
        # pad/truncate projection inputs
        if n_audio_embeds < audio_max:
            audio_embeds_raw = torch.concat(
                (torch.zeros(audio_max-n_audio_embeds,audio_C),audio_embeds_raw), # pad on left - this doesn't actually matter, I remove the padding later
                dim=0
            )
        else:
            audio_embeds_raw = audio_embeds_raw[:audio_max] # truncate on the right - this does matter.

        ret = {
            "attention_mask": attention_mask.squeeze(0),
            "input_embeds_raw": inputs_embeds_raw.squeeze(0),
            "input_ids_raw": input_ids_raw.squeeze(0),
            "audio_embeds_raw": audio_embeds_raw,
            "audio_tokens_start":torch.tensor(audio_tokens_start),
            "audio_tokens_end":torch.tensor(audio_tokens_end),
            "cap_tokens_start":torch.tensor(cap_tokens_start),
            "n_audio_embeds":torch.tensor(n_audio_embeds),
        }

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
