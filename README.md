# [EMNLP2024]Traffic Light or Light Traffic? Investigating Phrasal Semantics in Large Language Models

This is the official repository for the EMNLP 2024 paper [Traffic Light or Light Traffic? Investigating Phrasal Semantics in Large Language Models]([https://arxiv.org/abs/2109.06304](https://openreview.net/forum?id=5loBBDD3c3#discussion)). This repo is built based on [Phrase-BERT](https://github.com/sf-wa-326/phrase-bert-topic-model) and [PiC](https://github.com/Phrase-in-Context/eval).


Full results can be seen in this [Google Sheet](https://docs.google.com/spreadsheets/d/1LIog3UmaOw2sFsDQzgHdYfmX8Bqohy7mrriu87a-PPI/edit?usp=sharing).




### Command examples
Note: the models to run are configured in `config/model_path.py`.

#### Run OpenAI models on Turney/BiRD/PiC (4-shot and Chain-of-Thought prompt)
```bash
python eval_turney.py \
  --input_path data/turney/data_test2170.txt \
  --output_dir output/turney_test2170/ \ 
  --k_shot 4 --cot true --openai_key YOUR_KEY
```

```bash
python eval_bird.py \
    --input_path data/bird/data_test3335.txt \
    --output_dir output/bird_test3335/ \
    --k_shot 4 --cot true --openai_key YOUR_KEY
```

```bash
python eval_PiC_PS.py \
  --input_path data/PiC-PS/test-hard-v2.0.1.json \
  --output_dir output/PiC-PS_20240609/ \
  --k_shot 4 --cot true --openai_key YOUR_KEY
```