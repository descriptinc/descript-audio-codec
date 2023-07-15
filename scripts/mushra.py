import string
from dataclasses import dataclass
from pathlib import Path
from typing import List

import argbind
import gradio as gr
from audiotools import preference as pr


@argbind.bind(without_prefix=True)
@dataclass
class Config:
    folder: str = None
    save_path: str = "results.csv"
    conditions: List[str] = None
    reference: str = None
    seed: int = 0
    share: bool = False
    n_samples: int = 10


def get_text(wav_file: str):
    txt_file = Path(wav_file).with_suffix(".txt")
    if Path(txt_file).exists():
        with open(txt_file, "r") as f:
            txt = f.read()
    else:
        txt = ""
    return f"""<div style="text-align:center;font-size:large;">{txt}</div>"""


def main(config: Config):
    with gr.Blocks() as app:
        save_path = config.save_path
        samples = gr.State(pr.Samples(config.folder, n_samples=config.n_samples))

        reference = config.reference
        conditions = config.conditions

        player = pr.Player(app)
        player.create()
        if reference is not None:
            player.add("Play Reference")

        user = pr.create_tracker(app)
        ratings = []

        with gr.Row():
            txt = gr.HTML("")

        with gr.Row():
            gr.Button("Rate audio quality", interactive=False)
            with gr.Column(scale=8):
                gr.HTML(pr.slider_mushra)

        for i in range(len(conditions)):
            with gr.Row().style(equal_height=True):
                x = string.ascii_uppercase[i]
                player.add(f"Play {x}")
                with gr.Column(scale=9):
                    ratings.append(gr.Slider(value=50, interactive=True))

        def build(user, samples, *ratings):
            # Filter out samples user has done already, by looking in the CSV.
            samples.filter_completed(user, save_path)

            # Write results to CSV
            if samples.current > 0:
                start_idx = 1 if reference is not None else 0
                name = samples.names[samples.current - 1]
                result = {"sample": name, "user": user}
                for k, r in zip(samples.order[start_idx:], ratings):
                    result[k] = r
                pr.save_result(result, save_path)

            updates, done, pbar = samples.get_next_sample(reference, conditions)
            wav_file = updates[0]["value"]

            txt_update = gr.update(value=get_text(wav_file))

            return (
                updates
                + [gr.update(value=50) for _ in ratings]
                + [done, samples, pbar, txt_update]
            )

        progress = gr.HTML()
        begin = gr.Button("Submit", elem_id="start-survey")
        begin.click(
            fn=build,
            inputs=[user, samples] + ratings,
            outputs=player.to_list() + ratings + [begin, samples, progress, txt],
        ).then(None, _js=pr.reset_player)

        # Comment this back in to actually launch the script.
        app.launch(share=config.share)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        config = Config()
        main(config)
