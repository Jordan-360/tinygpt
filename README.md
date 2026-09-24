# tinygpt

A small GPT-style language model built and trained from scratch in PyTorch, running on a single consumer GPU (RTX 3070 Ti, 8GB).

No pretrained weights and no high-level model libraries: the transformer, training loop, and text generation are all implemented directly. Give it any plain-text file and it learns to write in that style.

Sample output after about 6 minutes of training on Shakespeare (prompt: `ROMEO:`):

```
ROMEO:
All thy brother's head and the duke a way
Of thy graves withal use a thing put of her back:
And all twice my father, when my father's counsel
When it now my best reems not with the cloud to the king.

GREMIO:
O God, lind of her husband with a cheeking head,
```

## Results

| Dataset | Size | Parameters | Hardware | Training time | Best val loss |
|---|---|---|---|---|---|
| Tiny Shakespeare | 1.1M chars | 10.8M | RTX 3070 Ti (8GB) | 5.8 min (5,000 steps) | 1.469 |

**Overfitting, visible in the chart below.** Validation loss reached its best of 1.469 and then rose to 1.508 by step 5,000, while training loss kept falling below 1.0. The model had started memorizing the training text instead of learning patterns that generalize. The script only saves the checkpoint with the best validation loss, so the saved model is the one from before overfitting set in.

<img width="1200" height="675" alt="shakespeare_loss" src="https://github.com/user-attachments/assets/926007fd-75fb-4eb2-b40e-b4207411ab91" />


**Learning in progress.** Samples from the same run at different points in training:
(venv) PS D:\Projects\tinygpt>    python tinygpt.py train --data input.txt --out models/shakespeare.pt
device=cuda | 1,115,394 chars | vocab=65 | params=10.8M
step     0 | train loss 4.330 | val loss 4.321 | 2s
   sample: '\nThU&;yQrcKidqMBYebEbHwPtROa!N3lv&LeGLtgblJQuH..cbTEOvObpLoq3EiyAky-vMcvyoKh3&:ObmhwrwePW-!S-Q-;!jrUuO,TQY!YAC.?\nYFM&!ljV'
step   500 | train loss 1.854 | val loss 1.975 | 36s
   sample: '\nQUENTILES:\nWhat oppet of these mptay that to suldy look\nSe I abjoy sove thant, morth sples with me so,\nHim to your, siul'
step  1000 | train loss 1.517 | val loss 1.713 | 71s
   sample: '\nDUKE VINCENTIO:\nPlow this I long this repent the counses\nA have in a well from to here, that the weath can of king I mar'
step  1500 | train loss 1.371 | val loss 1.590 | 105s
   sample: "\nAUFIUS:\nI at anger the woulder of your shopes our fortune,\nThis most are your body to the they body,\nSomen many consuit'"
step  2000 | train loss 1.286 | val loss 1.534 | 140s
   sample: '\n\nGLOUCESTER:\nI have been your heart.\n\nGLOUCESTER:\nThe king is late to know it, it is a cereman\nContemners shore the caus'
step  2500 | train loss 1.225 | val loss 1.498 | 174s
   sample: "\nAnd sure my honesty left and one in passing to\nsave my fault breath. He call'd me that a fearful and father\ntimes, hap r"
step  3000 | train loss 1.176 | val loss 1.485 | 210s
   sample: "\nCOMINLEO:\nMadam, I have, my master soul's all,\nAnd all the sons from the uther of their air,\nBanish'd by a mother pleasu"
step  3500 | train loss 1.126 | val loss 1.469 | 244s
   sample: "\nAnd have but my father's all be stones fault,\nShall you place me like in heaven, and we demase\nIn the common of my heart"
step  4000 | train loss 1.080 | val loss 1.490 | 278s
   sample: "\nI'll follow you.\n\nKING RICHARD II:\nMethought that I know the young of die.\n\nBUSHY:\nI take it not, hold is all my names s"
step  4500 | train loss 1.035 | val loss 1.480 | 314s
   sample: "\nWhere you noble Margaret's sons, that sensight,\nYou could confer the contrarge her from sheer\nBe freely well! My prayers"
step  5000 | train loss 0.998 | val loss 1.508 | 349s
   sample: "\nMaster are no hearing. Come, Lord of Signior France:\nI will be loathed on thee, soft I have convey'd\nAnd straitly by my "
done. best val loss 1.469 | model saved to models/shakespeare.pt | log saved to models/shakespeare_log.csv

**Temperature experiment.** The same model and prompt, sampled at three temperatures:

| Temperature | Excerpt | What changed |
|---|---|---|
| 0.5 | `Alas, for the world is the sea and the best / And that the day the sun that is well be.` | Mostly real words, but repetitive: ROMEO speaks four times in 300 characters. |
| 0.8 | `Of thy graves withal use a thing put of her back:` | A balance of real words, structure, and variety. |
| 1.2 | `Yonder Romeo, will you be but / backet boastard. / Stirrah, my liege.` | More creative and chaotic: invented words ("boastard," "achieful") and characters from different plays mixed together. |

Runs are reproducible: the script uses a fixed random seed, so two training runs with the same settings produced nearly identical results (best val loss 1.468 and 1.469).

## How it works

The model predicts the next character in a sequence, one character at a time. Everything it knows comes from repeating that task millions of times on the training text.

1. **Tokenizer:** each unique character in the dataset is mapped to an integer ID.
2. **Embeddings:** each ID, plus its position in the sequence, becomes a 384-dimensional vector.
3. **Transformer blocks (x6):** each block uses causal multi-head self-attention (6 heads), so every character can draw on the context before it, followed by a feed-forward network. Residual connections and layer normalization keep training stable.
4. **Output layer:** converts the final vectors into a probability for every possible next character.
5. **Training:** cross-entropy loss on next-character prediction, optimized with AdamW. Mixed-precision (bfloat16) training speeds things up on the GPU. The checkpoint with the best validation loss is kept.
6. **Generation:** the model samples a character, appends it, and repeats. A temperature setting controls how adventurous the sampling is.

| Setting | Value |
|---|---|
| Layers | 6 |
| Attention heads | 6 |
| Embedding size | 384 |
| Context length | 256 characters |
| Batch size | 64 |
| Learning rate | 3e-4 |
| Dropout | 0.2 |

## Quick start

Requires Python 3.10+ and a PyTorch install with CUDA ([pytorch.org](https://pytorch.org/get-started/locally/)). It also runs on CPU, just much more slowly.

```bash
git clone https://github.com/YOUR-USERNAME/tinygpt.git
cd tinygpt
pip install -r requirements.txt   # install torch with CUDA from pytorch.org first

# download the Tiny Shakespeare dataset
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# train (saves models/shakespeare.pt and a training log)
python tinygpt.py train --data input.txt --out models/shakespeare.pt

# generate text
python tinygpt.py generate --model models/shakespeare.pt --prompt "ROMEO:" --length 500

# plot the loss curve
python plot_loss.py models/shakespeare_log.csv --out assets/shakespeare_loss.png
```

Any UTF-8 text file works as training data. Around 1MB of text or more gives the best results; much smaller files tend to get memorized rather than learned.

## Project structure

```
tinygpt.py         model, training loop, and text generation (CLI)
plot_loss.py       turns a training log into a loss chart
assets/            charts and images used in this README
requirements.txt
```

## Acknowledgments

The architecture follows Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and his "Let's build GPT: from scratch, in code, spelled out" lecture. Tiny Shakespeare dataset from [char-rnn](https://github.com/karpathy/char-rnn).

## License

MIT
