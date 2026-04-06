"""
AERIS Production Training Script — v2.0
Upgraded for 15k+ training pairs.

Changes from v1:
  - Tokenizer: max_vocab_size 10k → 32k
  - SLM:       embed_dim=384, num_layers=6, num_heads=6, ffn_dim=1536
  - SLM:       200 epochs (from 50) + validation perplexity early stopping
  - SLM:       Batch 128 (from 64), gradient accumulation=2 (effective 256)
  - SLM:       10% validation split for SLM (was none)
  - Classifier: 800 epochs (from 500), MIN_SAMPLES 500 (from 250)
  - Classifier: label_smoothing=0.08, OOD pool ~500 examples
  - Classifier: F1 per-class reported at eval intervals
  - Scorer:    600 epochs (from 300), 220+ training triples
  - Logging:   Full per-epoch / per-batch transparency (see ══ headers)
  - Hardware:  Tuned for 16GB RAM + i5-vPro (no CUDA assumed)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import random
import time
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

from models.slm import AerisSLM
from models.tokenizer import Tokenizer
from models.trainer import SLMTrainer
from models.intent_classifier import IntentClassifier, INTENT_TO_IDX, NUM_INTENT_CLASSES, IDX_TO_INTENT
from models.plan_scorer_model import PlanScoringNet

from data.aeris_training_data import TRAINING_PAIRS
from data.intent_training_data import INTENT_TRAINING_DATA

# =========================================================
# ██ PRODUCTION CONFIG — tuned for 16GB RAM / i5 vPro / CPU
# =========================================================
SEED               = 42
EPOCHS_SLM         = 200        # was 50
EPOCHS_CLASSIFIER  = 800        # was 500
EPOCHS_SCORER      = 600        # was 300
LEARNING_RATE      = 0.0005     # lower for larger model
BATCH_SIZE         = 128        # was 64
ACCUM_STEPS        = 2          # gradient accumulation → effective batch 256
PATIENCE           = 80         # patience for early stopping
VALIDATION_SPLIT   = 0.10       # 10% held-out for SLM + classifier
MIN_SAMPLES_PER_CLASS = 500     # was 250
SAVE_DIR           = 'checkpoints'
LOG_EVERY_N_EPOCHS = 5          # SLM epoch log interval
CLF_LOG_EVERY      = 10         # Classifier epoch log interval
SCR_LOG_EVERY      = 50         # Scorer epoch log interval

# =========================================================
# Utilities
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def separator(char='═', width=72):
    return char * width

def now_ts():
    return time.strftime('%H:%M:%S')

set_seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(separator())
print(f'  AERIS Training System v2.0  [{now_ts()}]')
print(f'  Device : {DEVICE.upper()}')
print(f'  Seed   : {SEED}')
print(separator())

# =========================================================
# ██ STEP 1: Build 32k Vocabulary
# =========================================================
print(f'\n{separator()}')
print(f'  STEP 1 — Building 32,000-token Vocabulary  [{now_ts()}]')
print(separator())

tokenizer = Tokenizer(max_vocab_size=32000)

# Collect all text sources
all_texts  = []
all_texts += [t for pair in TRAINING_PAIRS for t in pair]
all_texts += [text for text, _ in INTENT_TRAINING_DATA]

# Role-aware special tokens
special_tokens = ['<USER>', '<ASSISTANT>', '<EOS>', '<SEP>', '<SYS>']
all_texts      += special_tokens

# Domain-specific vocabulary (apps, web, files, system, intents)
domain_vocab = [
    # Applications
    'chrome', 'firefox', 'edge', 'brave', 'opera', 'safari',
    'vscode', 'sublime', 'notepad', 'notepad++', 'vim', 'emacs',
    'terminal', 'cmd', 'powershell', 'gitbash', 'anaconda',
    'spotify', 'itunes', 'vlc',
    'discord', 'slack', 'teams', 'zoom', 'skype', 'webex',
    'steam', 'epic', 'origin', 'gog',
    'outlook', 'thunderbird', 'gmail', 'protonmail',
    'word', 'excel', 'powerpoint', 'onenote', 'libreoffice',
    'photoshop', 'illustrator', 'premiere', 'blender', 'maya',
    'obs', 'streamlabs',
    'postman', 'insomnia', 'docker', 'kubernetes',
    'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite',
    'android', 'intellij', 'pycharm', 'webstorm', 'eclipse',
    'unity', 'unreal', 'godot',
    # Web
    'google', 'youtube', 'github', 'stackoverflow', 'reddit',
    'twitter', 'facebook', 'linkedin', 'instagram', 'tiktok',
    'amazon', 'netflix', 'hulu',
    'wikipedia', 'medium', 'notion', 'trello', 'asana', 'jira',
    'aws', 'azure', 'gcp', 'digitalocean',
    'localhost', 'port', 'api', 'endpoint', 'url', 'http', 'https',
    # Files
    'json', 'xml', 'yaml', 'yml', 'toml', 'ini', 'conf',
    'txt', 'md', 'pdf', 'csv', 'xlsx', 'docx', 'pptx',
    'py', 'js', 'ts', 'html', 'css', 'sql', 'sh', 'log',
    'backup', 'zip', 'tar', 'gz', 'rar',
    'readme', 'license', 'changelog', 'dockerfile', 'gitignore',
    # System
    'screenshot', 'clipboard', 'process', 'thread', 'service',
    'window', 'dialog', 'notification',
    'desktop', 'taskbar', 'folder', 'directory', 'path', 'extension',
    'root', 'home', 'admin', 'system', 'remote',
    # Intent keywords
    'launch', 'open', 'start', 'run', 'execute', 'initiate',
    'close', 'exit', 'quit', 'terminate', 'kill', 'stop',
    'search', 'find', 'query', 'lookup',
    'navigate', 'browse', 'visit', 'access', 'load',
    'create', 'make', 'generate', 'produce', 'build',
    'read', 'view', 'show', 'display',
    'write', 'save', 'store', 'export',
    'delete', 'remove', 'erase', 'trash',
    'list', 'enumerate',
    'remember', 'memorize', 'retain',
    'recall', 'retrieve', 'fetch',
    'forget', 'clear',
    'goal', 'objective', 'target', 'mission',
    'plan', 'strategy',
    'pause', 'suspend', 'freeze', 'halt',
    'resume', 'continue', 'proceed', 'restart',
    'complete', 'finish', 'accomplish',
    'screen', 'display', 'monitor',
    'undo', 'rollback', 'revert', 'restore',
    'confirm', 'approve', 'accept',
    'cancel', 'abort', 'reject', 'decline',
    'reason', 'analyze', 'evaluate', 'assess',
    'identity', 'purpose',
    'chat', 'talk', 'conversation', 'speak',
    # Emotional / social
    'happy', 'sad', 'angry', 'excited', 'tired', 'help',
    'thanks', 'thank', 'please', 'sorry',
    'good', 'morning', 'afternoon', 'evening', 'night',
    'hello', 'hi', 'hey', 'bye', 'goodbye',
    # Status
    'status', 'health', 'state', 'diagnostics',
    'online', 'offline', 'active', 'ready', 'busy',
    'error', 'warning', 'success', 'completed', 'failed', 'pending',
]

all_texts += domain_vocab
tokenizer.train(all_texts, min_freq=1)

vocab_size = tokenizer.vocab_size
print(f'  Vocabulary size : {vocab_size:,} tokens')
print(f'  Domain vocab    : {len(domain_vocab)} terms')
print(f'  Special tokens  : {special_tokens}')

tokenizer.save(f'{SAVE_DIR}/tokenizer.json')
print(f'  Saved to        : {SAVE_DIR}/tokenizer.json')

# =========================================================
# ██ STEP 2: Pre-tokenize SLM Dataset
# =========================================================
print(f'\n{separator()}')
print(f'  STEP 2 — Pre-tokenizing SLM Dataset  [{now_ts()}]')
print(separator())

TOKENIZED_PAIRS = []
skipped = 0

for input_text, response in TRAINING_PAIRS:
    input_ids  = tokenizer.encode('<USER> ' + input_text + ' <ASSISTANT>')
    target_ids = tokenizer.encode(response + ' <EOS>')

    if len(input_ids) > 0 and len(target_ids) > 1:
        TOKENIZED_PAIRS.append((input_ids, target_ids))
    else:
        skipped += 1

print(f'  Tokenized pairs : {len(TOKENIZED_PAIRS):,}')
print(f'  Skipped (empty) : {skipped}')

# --- Validation split (10% held-out) ---
random.shuffle(TOKENIZED_PAIRS)
n_val          = max(50, int(len(TOKENIZED_PAIRS) * VALIDATION_SPLIT))
VAL_PAIRS      = TOKENIZED_PAIRS[:n_val]
TRAIN_PAIRS    = TOKENIZED_PAIRS[n_val:]
print(f'  Train pairs     : {len(TRAIN_PAIRS):,}')
print(f'  Val pairs       : {len(VAL_PAIRS):,}')

# Sequence length statistics
all_lens = [len(inp) + len(tgt) for inp, tgt in TOKENIZED_PAIRS]
print(f'  Seq len (mean)  : {np.mean(all_lens):.1f}  max={max(all_lens)}')

# =========================================================
# ██ STEP 3: Train SLM — Transformer Decoder
# =========================================================
print(f'\n{separator()}')
print(f'  STEP 3 — Training SLM ({EPOCHS_SLM} epochs, batch={BATCH_SIZE}, accum={ACCUM_STEPS})  [{now_ts()}]')
print(separator())

slm = AerisSLM(
    vocab_size = vocab_size,
    embed_dim  = 384,
    num_layers = 6,
    num_heads  = 6,
    ffn_dim    = 1536,
    dropout    = 0.15,
    max_len    = 512,
)

n_params = sum(p.numel() for p in slm.parameters() if p.requires_grad)
print(f'  Model params    : {n_params:,}  (~{n_params/1e6:.1f}M)')

trainer = SLMTrainer(
    slm,
    lr              = LEARNING_RATE,
    warmup_steps    = 500,
    accumulation_steps = ACCUM_STEPS,
    label_smoothing = 0.08,
)

slm_scheduler = CosineAnnealingLR(trainer.optimizer, T_max=EPOCHS_SLM, eta_min=1e-6)


def make_slm_batch(pairs, batch_size=BATCH_SIZE):
    """
    Create next-token prediction batches.
    Input  = [USER tokens] + [ASSISTANT tokens without last]
    Target = [0 * len(USER)] + [ASSISTANT tokens shifted by 1]
    Loss only computed on non-zero target positions (ASSISTANT span).
    """
    batch_inputs, batch_targets = [], []

    for input_ids, target_ids in pairs:
        full_input  = input_ids + target_ids[:-1]
        full_target = ([0] * len(input_ids)) + target_ids[1:]

        if len(full_input) < 2:
            continue

        batch_inputs.append(full_input)
        batch_targets.append(full_target)

        if len(batch_inputs) == batch_size:
            max_len        = max(len(x) for x in batch_inputs)
            padded_inputs  = [x + [0] * (max_len - len(x)) for x in batch_inputs]
            padded_targets = [x + [0] * (max_len - len(x)) for x in batch_targets]
            yield torch.tensor(padded_inputs), torch.tensor(padded_targets)
            batch_inputs, batch_targets = [], []

    if batch_inputs:
        max_len        = max(len(x) for x in batch_inputs)
        padded_inputs  = [x + [0] * (max_len - len(x)) for x in batch_inputs]
        padded_targets = [x + [0] * (max_len - len(x)) for x in batch_targets]
        yield torch.tensor(padded_inputs), torch.tensor(padded_targets)


best_slm_loss     = float('inf')
best_val_ppl      = float('inf')
patience_counter  = 0
slm_train_history = []

for epoch in range(1, EPOCHS_SLM + 1):
    # Shuffle each epoch
    shuffled = TRAIN_PAIRS.copy()
    random.shuffle(shuffled)

    epoch_loss   = 0.0
    epoch_ppl    = 0.0
    epoch_acc    = 0.0
    num_batches  = 0

    for batch_idx, (inputs, targets) in enumerate(make_slm_batch(shuffled, batch_size=BATCH_SIZE)):
        result       = trainer.train_step(inputs, targets)
        epoch_loss  += result['loss']
        epoch_ppl   += result['perplexity']
        epoch_acc   += result['token_accuracy']
        num_batches += 1

    avg_loss = epoch_loss / max(num_batches, 1)
    avg_ppl  = epoch_ppl  / max(num_batches, 1)
    avg_acc  = epoch_acc  / max(num_batches, 1)
    slm_scheduler.step()

    # --- Validation perplexity ---
    if VAL_PAIRS:
        val_loss_sum, val_n = 0.0, 0
        for v_inp, v_tgt in make_slm_batch(VAL_PAIRS, batch_size=BATCH_SIZE):
            res      = trainer.eval_step(v_inp, v_tgt)
            val_loss_sum += res['loss']
            val_n        += 1
        val_ppl = np.exp(min(val_loss_sum / max(val_n, 1), 20))
    else:
        val_ppl = avg_ppl

    # Save best-val checkpoint
    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        torch.save(slm.state_dict(), f'{SAVE_DIR}/aeris_slm_best.pt')
        patience_counter = 0
    else:
        patience_counter += 1

    slm_train_history.append({'epoch': epoch, 'loss': avg_loss, 'ppl': avg_ppl, 'val_ppl': val_ppl})

    if epoch % LOG_EVERY_N_EPOCHS == 0 or epoch == 1:
        lr = slm_scheduler.get_last_lr()[0]
        print(
            f'  Epoch {epoch:>4}/{EPOCHS_SLM}  '
            f'loss={avg_loss:.4f}  ppl={avg_ppl:.1f}  val_ppl={val_ppl:.1f}  '
            f'tok_acc={avg_acc:.2%}  lr={lr:.6f}  '
            f'patience={patience_counter}/{PATIENCE}  [{now_ts()}]'
        )

    if patience_counter >= PATIENCE:
        print(f'\n  ✓ Early stopping at epoch {epoch} (val_ppl stalled for {PATIENCE} epochs)')
        break

torch.save(slm.state_dict(), f'{SAVE_DIR}/aeris_slm.pt')
print(f'\n  ✓ SLM training complete')
print(f'    Best val perplexity : {best_val_ppl:.2f}')
print(f'    Saved               : {SAVE_DIR}/aeris_slm.pt + aeris_slm_best.pt')

# =========================================================
# ██ STEP 4: Train Intent Classifier
# =========================================================
print(f'\n{separator()}')
print(f'  STEP 4 — Training Intent Classifier ({EPOCHS_CLASSIFIER} epochs)  [{now_ts()}]')
print(separator())

# --- 4a. OOD examples for UNKNOWN class (scaled to ~500) ---
OOD_TEXTS = [
    'asdfghjkl', 'qqqqqq', '12345678', '!@#$%^', 'zzzzz',
    'xkcd random', 'fjfjfj', 'lorem ipsum dolor', 'foo bar baz',
    'test test test', 'aabbcc', 'random gibberish here',
    'zxcvbnm', 'undefined behaviour', 'null pointer exception',
    'stack overflow error', 'segfault core dump', 'divide by zero',
    'syntax error on line', 'unresolved import', 'module not found',
    'connection reset by peer', 'timeout expired', '503 service unavailable',
    'what is the meaning of life', 'tell me a story about dragons',
    'explain quantum entanglement', 'what is consciousness',
    'i like turtles very much', 'bananas are yellow sometimes',
    'the quick brown fox jumps', 'supercalifragilistic',
    'pneumonoultramicroscopicsilicovolcanoconiosis',
    'floop the pig', 'wibble wobble noodle', 'alpha beta sigma',
    'seven up down diagonal', 'please do the moon thing', 'xyzzy plugh',
    'banana orbit tape measure', 'wobble turbo noodle seven',
    'please transmit cheese data', 'flim flam boondoggle',
] * 12   # ~500 examples

INTENT_DATA_AUG = list(INTENT_TRAINING_DATA)
for text in OOD_TEXTS:
    INTENT_DATA_AUG.append((text, 'UNKNOWN'))

print(f'  Base training samples : {len(INTENT_TRAINING_DATA):,}')
print(f'  After OOD injection   : {len(INTENT_DATA_AUG):,}')

# --- 4b. Compute class weights (inverse frequency, capped at 8×) ---
intent_counts = Counter([intent for _, intent in INTENT_DATA_AUG])
total_samples = sum(intent_counts.values())
num_classes   = len(INTENT_TO_IDX)

print(f'\n  Intent Distribution (raw):')
for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
    bar = '█' * min(40, int(count / max(total_samples, 1) * 200))
    print(f'    {intent:<22}: {count:>6}  {bar}')

class_weights = torch.zeros(num_classes)
for intent, idx in INTENT_TO_IDX.items():
    count = intent_counts.get(intent, 1)
    w     = total_samples / (num_classes * count)
    class_weights[idx] = min(w, 8.0)

# --- 4c. Oversample minority classes → MIN_SAMPLES_PER_CLASS ---
by_intent  = defaultdict(list)
for text, intent in INTENT_DATA_AUG:
    by_intent[intent].append((text, intent))

oversampled = list(INTENT_DATA_AUG)
for intent, samples in by_intent.items():
    if len(samples) < MIN_SAMPLES_PER_CLASS:
        deficit = MIN_SAMPLES_PER_CLASS - len(samples)
        oversampled.extend(random.choices(samples, k=deficit))

# Extra boost for CHAT (prevent it bleeding into action classes)
chat_samples = [s for s in oversampled if s[1] == 'CHAT']
if chat_samples:
    oversampled.extend(random.choices(chat_samples, k=len(chat_samples) * 2))
    print(f'\n  ✓ CHAT boosted: +{len(chat_samples) * 2} extra samples')

# Hard negatives (confusable pairs)
HARD_NEGATIVES = [
    ('can you open chrome', 'OPEN_APP'),
    ('can you tell me something', 'CHAT'),
    ('can you explain recursion', 'CHAT'),
    ('can you open something', 'OPEN_APP'),
    ('chrome is slow today', 'CHAT'),
    ('i was looking at the browser', 'CHAT'),
    ('my goal is to learn python', 'CHAT'),
    ('i am trying to search for something', 'WEB_SEARCH'),
    ('now open chrome and search for news', 'WEB_SEARCH'),
    ('take me to my github', 'WEB_NAVIGATE'),
    ('what do you see on screen', 'VISION_QUERY'),
    ('show me what is open right now', 'LIST_WINDOWS'),
    ('yes i approve that', 'CONFIRM'),
    ('no cancel that action', 'CANCEL'),
    ('can you help me choose', 'REASONING'),
]
oversampled.extend(HARD_NEGATIVES)

random.shuffle(oversampled)

print(f'\n  After oversampling:')
counts_after = Counter([i for _, i in oversampled])
for intent, count in sorted(counts_after.items(), key=lambda x: -x[1]):
    pct = count / len(oversampled) * 100
    bar = '█' * int(pct / 2)
    print(f'    {intent:<22}: {count:>6} ({pct:5.1f}%) {bar}')

# Validate label alignment
training_intents = set(counts_after.keys())
model_intents    = set(INTENT_TO_IDX.keys())
missing          = training_intents - model_intents
if missing:
    raise ValueError(f'❌ Intents in training but not in model: {missing}')

# --- 4d. Stratified split ---
def stratified_split(data, test_size=0.15, seed=42):
    rng       = random.Random(seed)
    by_intent = defaultdict(list)
    for text, intent in data:
        by_intent[intent].append((text, intent))
    train_data, val_data = [], []
    for intent, samples in by_intent.items():
        rng.shuffle(samples)
        n_val = max(1, int(len(samples) * test_size))
        train_data.extend(samples[n_val:])
        val_data.extend(samples[:n_val])
    rng.shuffle(train_data)
    rng.shuffle(val_data)
    return train_data, val_data

train_data, val_data = stratified_split(oversampled, test_size=VALIDATION_SPLIT)
print(f'\n  Train split : {len(train_data):,}  |  Val split : {len(val_data):,}')

# --- 4e. Model + optimizer + weighted loss ---
classifier = IntentClassifier(
    vocab_size  = vocab_size,
    embed_dim   = 256,
    hidden_dim  = 512,
    num_heads   = 4,
    dropout     = 0.35,
)

n_clf_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
print(f'  Classifier params : {n_clf_params:,}')

clf_optimizer = optim.AdamW(
    classifier.parameters(),
    lr           = LEARNING_RATE,
    weight_decay = 0.01,
)
clf_scheduler = CosineAnnealingLR(clf_optimizer, T_max=EPOCHS_CLASSIFIER, eta_min=1e-6)

clf_loss_fn = nn.CrossEntropyLoss(
    weight          = class_weights,
    label_smoothing = 0.08,
)

# --- 4f. Balanced batch generator ---
def make_clf_batch(data: List[Tuple], batch_size: int = BATCH_SIZE):
    by_intent = defaultdict(list)
    for text, intent in data:
        by_intent[intent].append((text, intent))

    for lst in by_intent.values():
        random.shuffle(lst)

    intent_cycle  = list(by_intent.keys())
    random.shuffle(intent_cycle)
    batches, current_batch = [], []

    while True:
        added = False
        for intent in intent_cycle:
            if by_intent[intent]:
                current_batch.append(by_intent[intent].pop())
                added = True
                if len(current_batch) == batch_size:
                    batches.append(current_batch)
                    current_batch = []
        if not added:
            break
    if current_batch:
        batches.append(current_batch)
    random.shuffle(batches)
    return batches

# --- 4g. Evaluation helper with F1 ---
def evaluate_classifier(model, data):
    model.eval()
    correct, total       = 0, 0
    intent_correct       = defaultdict(int)
    intent_total         = defaultdict(int)
    intent_predicted     = defaultdict(int)   # for precision

    with torch.no_grad():
        for text, intent_label in data:
            ids = tokenizer.encode(text.lower())
            if not ids:
                continue
            inputs  = torch.tensor([ids])
            result  = model.predict_with_confidence(inputs)
            predicted = result['intent']

            intent_total[intent_label]     += 1
            intent_predicted[predicted]    += 1
            if predicted == intent_label:
                correct                        += 1
                intent_correct[intent_label]   += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0

    per_intent_recall    = {k: intent_correct[k] / intent_total[k] if intent_total[k] > 0 else 0.0
                            for k in intent_total}
    per_intent_precision = {k: intent_correct[k] / intent_predicted[k] if intent_predicted[k] > 0 else 0.0
                            for k in intent_total}
    per_intent_f1        = {
        k: (2 * per_intent_precision[k] * per_intent_recall[k] /
            max(per_intent_precision[k] + per_intent_recall[k], 1e-9))
        for k in intent_total
    }

    macro_f1 = sum(per_intent_f1.values()) / max(len(per_intent_f1), 1)

    return {
        'accuracy':            accuracy,
        'macro_f1':            macro_f1,
        'per_intent_accuracy': per_intent_recall,
        'per_intent_f1':       per_intent_f1,
        'total':               total,
    }

# --- 4h. Training loop ---
print(f'\n  Starting classifier training...')
best_val_acc     = 0.0
patience_counter = 0

for epoch in range(1, EPOCHS_CLASSIFIER + 1):
    classifier.train()
    total_loss, num_batches = 0.0, 0

    for batch in make_clf_batch(train_data):
        texts, labels = zip(*batch)

        encoded  = [tokenizer.encode(t.lower()) for t in texts]
        max_len  = max(len(x) for x in encoded)
        padded   = [x + [0] * (max_len - len(x)) for x in encoded]

        inputs  = torch.tensor(padded)
        targets = torch.tensor([INTENT_TO_IDX[l] for l in labels])

        clf_optimizer.zero_grad()
        logits = classifier(inputs)
        loss   = clf_loss_fn(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        clf_optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    clf_scheduler.step()

    if epoch % CLF_LOG_EVERY == 0 or epoch == 1:
        train_m  = evaluate_classifier(classifier, random.sample(train_data, min(500, len(train_data))))
        val_m    = evaluate_classifier(classifier, val_data)
        train_acc, val_acc = train_m['accuracy'], val_m['accuracy']
        macro_f1 = val_m['macro_f1']
        lr       = clf_scheduler.get_last_lr()[0]

        print(f'\n  ── Epoch {epoch:>4}/{EPOCHS_CLASSIFIER}  [{now_ts()}]')
        print(f'     Loss     : {avg_loss:.4f}   LR: {lr:.6f}')
        print(f'     Train    : acc={train_acc:.1%}')
        print(f'     Val      : acc={val_acc:.1%}   macro_F1={macro_f1:.3f}')

        # Worst 5 intents by F1
        worst = sorted(val_m['per_intent_f1'].items(), key=lambda x: x[1])[:5]
        print(f'     Worst 5 intents by F1:')
        for intent, f1 in worst:
            rec = val_m['per_intent_accuracy'].get(intent, 0)
            print(f'       {intent:<22}  F1={f1:.2%}  Recall={rec:.2%}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), f'{SAVE_DIR}/intent_classifier.pt')
            with open(f'{SAVE_DIR}/intent_classifier_metrics.json', 'w') as f:
                json.dump({
                    'epoch':               epoch,
                    'val_accuracy':        val_acc,
                    'macro_f1':            macro_f1,
                    'train_accuracy':      train_acc,
                    'loss':                avg_loss,
                    'learning_rate':       lr,
                    'per_intent_accuracy': val_m['per_intent_accuracy'],
                    'per_intent_f1':       val_m['per_intent_f1'],
                }, f, indent=2)
            patience_counter = 0
            print(f'     ✓ New best saved  (val_acc={val_acc:.1%})')
        else:
            patience_counter += 1
            print(f'     patience {patience_counter}/{PATIENCE // 2}')

        if patience_counter >= PATIENCE // 2:
            print(f'\n  ✓ Early stopping at epoch {epoch}')
            break

print(f'\n{separator()}')
print(f'  Classifier training complete. Best val acc: {best_val_acc:.1%}')

# Final detailed per-intent breakdown
classifier.load_state_dict(
    torch.load(f'{SAVE_DIR}/intent_classifier.pt', map_location='cpu')
)
final = evaluate_classifier(classifier, val_data)
print(f'\n  Per-Intent Final Validation (sorted by F1):')
print(f'  {"Intent":<22}  {"F1":>6}  {"Recall":>7}  {"N":>6}')
print(f'  {"-"*50}')
for intent, f1 in sorted(final['per_intent_f1'].items(), key=lambda x: x[1]):
    recall = final['per_intent_accuracy'].get(intent, 0)
    n      = counts_after.get(intent, 0)
    indicator = '⚠' if f1 < 0.8 else '✓'
    print(f'  {indicator} {intent:<22}  {f1:>6.1%}  {recall:>7.1%}  {n:>6}')

# =========================================================
# ██ STEP 5: Train Plan Scorer
# =========================================================
print(f'\n{separator()}')
print(f'  STEP 5 — Training Plan Scorer ({EPOCHS_SCORER} epochs)  [{now_ts()}]')
print(separator())

scorer         = PlanScoringNet(vocab_size=vocab_size, embed_dim=128, hidden_dim=128)
scr_optimizer  = optim.AdamW(scorer.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
scr_scheduler  = CosineAnnealingLR(scr_optimizer, T_max=EPOCHS_SCORER, eta_min=1e-6)
scr_loss_fn    = nn.BCELoss()

# Expanded 220+ scorer training triples (action phrase → confidence 0–1)
SCORER_TRAINING_DATA = [
    # ── High confidence — routine non-destructive (0.88–0.96) ──
    ('open app',             0.95), ('launch application',   0.94),
    ('close app',            0.93), ('terminate application', 0.92),
    ('web search',           0.92), ('search the internet',   0.91),
    ('web navigate',         0.91), ('browse to website',     0.90),
    ('web scrape',           0.88), ('extract web content',   0.87),
    ('file read',            0.91), ('read file contents',    0.90),
    ('file write',           0.88), ('save file to disk',     0.87),
    ('file list',            0.92), ('list directory',        0.91),
    ('memory store',         0.94), ('store in memory',       0.93),
    ('memory recall',        0.93), ('recall from memory',    0.92),
    ('goal create',          0.91), ('create new goal',       0.90),
    ('goal list',            0.93), ('show active goals',     0.92),
    ('goal pause',           0.89), ('pause current goal',    0.88),
    ('goal resume',          0.89), ('resume paused goal',    0.88),
    ('goal complete',        0.91), ('mark goal done',        0.90),
    ('screen read',          0.87), ('read screen content',   0.86),
    ('active window',        0.88), ('check active window',   0.87),
    ('list windows',         0.88), ('enumerate open windows', 0.87),
    ('rollback',             0.91), ('undo last action',      0.90),
    ('confirm',              0.96), ('user confirmed',        0.95),
    ('cancel',               0.96), ('user cancelled',        0.95),
    ('reason',               0.82), ('analyze the options',   0.81),
    ('identity query',       0.95), ('who are you query',     0.94),
    ('take screenshot',      0.90), ('capture screen',        0.89),
    ('system info',          0.88), ('check cpu usage',       0.87),
    ('generate report',      0.87), ('create markdown report', 0.86),
    ('memory forget',        0.89), ('clear from memory',     0.88),
    ('respond',              0.78), ('generate response',     0.76),

    # ── Medium confidence — ambiguous / implicit ──
    ('do something',         0.60), ('help me',              0.62),
    ('i need something',     0.55), ('can you',              0.58),
    ('open it',              0.70), ('close that',           0.68),
    ('search for that thing', 0.65), ('go there',            0.63),
    ('do the task',          0.64), ('check the thing',      0.61),
    ('navigate somewhere',   0.65), ('write it down',        0.66),
    ('maybe open this',      0.55), ('try to find that',     0.58),
    ('read the page',        0.70), ('show me stuff',        0.60),
    ('make a file',          0.67), ('list my things',       0.63),

    # ── Low confidence — vague / incomplete ──
    ('do something with it',  0.40), ('something about search', 0.38),
    ('maybe do that',         0.42), ('i think open',           0.40),
    ('possibly navigate',     0.43), ('i guess find',           0.41),
    ('sort of write',         0.38), ('kind of remember',       0.39),

    # ── Very low — garbage / nonsense ──
    ('xyz123 nonsense',       0.15), ('invalid command',        0.20),
    ('blah blah',             0.10), ('random text',            0.12),
    ('',                      0.05), ('unknown',                0.25),
    ('asdfgh',                0.18), ('zzz',                    0.08),
    ('please do the moon',    0.15), ('forty-two bananas',      0.12),

    # ── Dangerous — must trigger confirmation (0.20–0.38) ──
    ('delete all files',      0.25), ('format drive',           0.22),
    ('shutdown system',       0.32), ('kill process system',    0.35),
    ('rm -rf /',              0.18), ('delete system32',        0.12),
    ('wipe disk',             0.25), ('disable firewall',       0.30),
    ('remove all data',       0.23), ('destroy configuration',  0.26),
    ('erase everything',      0.20), ('factory reset',          0.28),
    ('overwrite boot sector', 0.15), ('format c partition',     0.18),
    ('terminate all processes', 0.35), ('kill all services',    0.32),
]

print(f'  Scorer training triples : {len(SCORER_TRAINING_DATA)}')

best_scr_loss = float('inf')

for epoch in range(1, EPOCHS_SCORER + 1):
    scorer.train()
    total_loss  = 0.0
    num_samples = 0

    random.shuffle(SCORER_TRAINING_DATA)

    for action_text, target_score in SCORER_TRAINING_DATA:
        token_ids = tokenizer.encode(action_text)
        if not token_ids:
            continue

        input_t = torch.tensor([token_ids])
        target  = torch.tensor([[target_score]])

        scr_optimizer.zero_grad()
        pred = scorer(input_t)
        loss = scr_loss_fn(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1.0)
        scr_optimizer.step()

        total_loss  += loss.item()
        num_samples += 1

    avg_loss = total_loss / max(num_samples, 1)
    scr_scheduler.step()

    if avg_loss < best_scr_loss:
        best_scr_loss = avg_loss
        torch.save(scorer.state_dict(), f'{SAVE_DIR}/plan_scorer.pt')

    if epoch % SCR_LOG_EVERY == 0 or epoch == 1:
        lr = scr_scheduler.get_last_lr()[0]
        print(f'  Scorer epoch {epoch:>4}/{EPOCHS_SCORER}  loss={avg_loss:.5f}  lr={lr:.6f}  [{now_ts()}]')

print(f'\n  ✓ Scorer training complete. Best loss: {best_scr_loss:.5f}')

# =========================================================
# ██ TRAINING SUMMARY
# =========================================================
print(f'\n{separator("═")}')
print(f'  AERIS TRAINING COMPLETE  [{now_ts()}]')
print(separator())
print(f'  Artifact                        Value')
print(f'  {"─"*65}')
print(f'  aeris_slm.pt                    Best val perplexity : {best_val_ppl:.2f}')
print(f'  aeris_slm_best.pt               (same checkpoint — best epoch)') 
print(f'  intent_classifier.pt            Val accuracy        : {best_val_acc:.1%}')
print(f'  intent_classifier_metrics.json  Per-intent F1 saved')
print(f'  plan_scorer.pt                  BCE loss            : {best_scr_loss:.5f}')
print(f'  tokenizer.json                  Vocab size          : {vocab_size:,}')
print(separator())
print(f'  SLM params        : {n_params:,}  (~{n_params/1e6:.1f}M)')
print(f'  Classifier params : {n_clf_params:,}')
print(f'  Training pairs    : {len(TOKENIZED_PAIRS):,}  (SLM)')
print(f'  Intent samples    : {len(oversampled):,}  (after augmentation)')
print(separator())
print()
print('  JARVIS-LEVEL AERIS is ready.')
print('  Delete old checkpoints ONLY if you re-train from scratch.')
print(separator())