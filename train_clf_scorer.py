"""
AERIS Classifier + Scorer Training — Fast Resume Script
Skips SLM (already trained). Loads existing tokenizer.json.
Run this after stopping train.py early.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import random
import time
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter, defaultdict
from typing import List, Tuple

from models.tokenizer import Tokenizer
from models.intent_classifier import IntentClassifier, INTENT_TO_IDX, NUM_INTENT_CLASSES, IDX_TO_INTENT
from models.plan_scorer_model import PlanScoringNet

from data.intent_training_data import INTENT_TRAINING_DATA

# =========================================================
# Config — same as train.py
# =========================================================
SEED                  = 42
EPOCHS_CLASSIFIER     = 800
EPOCHS_SCORER         = 600
LEARNING_RATE         = 0.0005
BATCH_SIZE            = 128
PATIENCE              = 80
VALIDATION_SPLIT      = 0.10
MIN_SAMPLES_PER_CLASS = 500
SAVE_DIR              = 'checkpoints'
CLF_LOG_EVERY         = 10
SCR_LOG_EVERY         = 50

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def separator(char='═', width=72):
    return char * width

def now_ts():
    return time.strftime('%H:%M:%S')

set_seed(SEED)

print(separator())
print(f'  AERIS Classifier + Scorer Training  [{now_ts()}]')
print(f'  (SLM already trained — loading tokenizer from checkpoint)')
print(separator())

# =========================================================
# Load tokenizer from checkpoint (no rebuild needed)
# =========================================================
TOKENIZER_PATH = f'{SAVE_DIR}/tokenizer.json'
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f'tokenizer.json not found at {TOKENIZER_PATH}. Run train.py first.')

tokenizer = Tokenizer(max_vocab_size=32000)
tokenizer.load(TOKENIZER_PATH)
vocab_size = tokenizer.vocab_size
print(f'  Tokenizer loaded: {vocab_size:,} tokens\n')

# =========================================================
# STEP 1: Train Intent Classifier
# =========================================================
print(separator())
print(f'  STEP 1 — Training Intent Classifier ({EPOCHS_CLASSIFIER} epochs)  [{now_ts()}]')
print(separator())

# --- OOD injection ---
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
] * 12

INTENT_DATA_AUG = list(INTENT_TRAINING_DATA)
for text in OOD_TEXTS:
    INTENT_DATA_AUG.append((text, 'UNKNOWN'))

print(f'  Base samples  : {len(INTENT_TRAINING_DATA):,}')
print(f'  After OOD     : {len(INTENT_DATA_AUG):,}')

# --- Class weights ---
intent_counts = Counter([intent for _, intent in INTENT_DATA_AUG])
total_samples = sum(intent_counts.values())
num_classes   = len(INTENT_TO_IDX)

print(f'\n  Intent Distribution:')
for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
    bar = '█' * min(40, int(count / max(total_samples, 1) * 200))
    print(f'    {intent:<22}: {count:>6}  {bar}')

class_weights = torch.zeros(num_classes)
for intent, idx in INTENT_TO_IDX.items():
    count = intent_counts.get(intent, 1)
    w     = total_samples / (num_classes * count)
    class_weights[idx] = min(w, 8.0)

# --- Oversample minority classes ---
by_intent  = defaultdict(list)
for text, intent in INTENT_DATA_AUG:
    by_intent[intent].append((text, intent))

oversampled = list(INTENT_DATA_AUG)
for intent, samples in by_intent.items():
    if len(samples) < MIN_SAMPLES_PER_CLASS:
        deficit = MIN_SAMPLES_PER_CLASS - len(samples)
        oversampled.extend(random.choices(samples, k=deficit))

chat_samples = [s for s in oversampled if s[1] == 'CHAT']
if chat_samples:
    oversampled.extend(random.choices(chat_samples, k=len(chat_samples) * 2))
    print(f'\n  CHAT boosted: +{len(chat_samples) * 2} extra samples')

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

training_intents = set(counts_after.keys())
model_intents    = set(INTENT_TO_IDX.keys())
missing          = training_intents - model_intents
if missing:
    raise ValueError(f'Intent mismatch: {missing}')

# --- Stratified split ---
def stratified_split(data, test_size=0.15, seed=42):
    rng = random.Random(seed)
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
print(f'\n  Train : {len(train_data):,}  |  Val : {len(val_data):,}')

# --- Model ---
classifier = IntentClassifier(
    vocab_size  = vocab_size,
    embed_dim   = 256,
    hidden_dim  = 512,
    num_heads   = 4,
    dropout     = 0.35,
)
n_clf = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
print(f'  Classifier params : {n_clf:,}')

clf_optimizer = optim.AdamW(classifier.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
clf_scheduler = CosineAnnealingLR(clf_optimizer, T_max=EPOCHS_CLASSIFIER, eta_min=1e-6)
clf_loss_fn   = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.08)

# --- Balanced batch generator ---
def make_clf_batch(data: List[Tuple], batch_size: int = BATCH_SIZE):
    by_intent = defaultdict(list)
    for text, intent in data:
        by_intent[intent].append((text, intent))
    for lst in by_intent.values():
        random.shuffle(lst)
    intent_cycle = list(by_intent.keys())
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

# --- Evaluation with F1 ---
def evaluate_classifier(model, data):
    model.eval()
    correct, total           = 0, 0
    intent_correct           = defaultdict(int)
    intent_total             = defaultdict(int)
    intent_predicted         = defaultdict(int)

    with torch.no_grad():
        for text, intent_label in data:
            ids = tokenizer.encode(text.lower())
            if not ids:
                continue
            inputs    = torch.tensor([ids])
            result    = model.predict_with_confidence(inputs)
            predicted = result['intent']
            intent_total[intent_label]  += 1
            intent_predicted[predicted] += 1
            if predicted == intent_label:
                correct += 1
                intent_correct[intent_label] += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    per_recall    = {k: intent_correct[k] / intent_total[k] if intent_total[k] > 0 else 0.0 for k in intent_total}
    per_precision = {k: intent_correct[k] / intent_predicted[k] if intent_predicted[k] > 0 else 0.0 for k in intent_total}
    per_f1        = {
        k: (2 * per_precision[k] * per_recall[k] / max(per_precision[k] + per_recall[k], 1e-9))
        for k in intent_total
    }
    macro_f1 = sum(per_f1.values()) / max(len(per_f1), 1)
    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'per_intent_accuracy': per_recall, 'per_intent_f1': per_f1, 'total': total}

# --- Training loop ---
print(f'\n  Starting classifier training...\n')
best_val_acc     = 0.0
patience_counter = 0

for epoch in range(1, EPOCHS_CLASSIFIER + 1):
    classifier.train()
    total_loss, num_batches = 0.0, 0

    for batch in make_clf_batch(train_data):
        texts, labels = zip(*batch)
        encoded = [tokenizer.encode(t.lower()) for t in texts]
        max_len = max(len(x) for x in encoded)
        padded  = [x + [0] * (max_len - len(x)) for x in encoded]
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

        print(f'  ── Epoch {epoch:>4}/{EPOCHS_CLASSIFIER}  [{now_ts()}]')
        print(f'     Loss: {avg_loss:.4f}  LR: {lr:.6f}')
        print(f'     Train: {train_acc:.1%}  |  Val: {val_acc:.1%}  |  Macro F1: {macro_f1:.3f}')

        worst = sorted(val_m['per_intent_f1'].items(), key=lambda x: x[1])[:5]
        print(f'     Worst 5 by F1:')
        for intent, f1 in worst:
            rec = val_m['per_intent_accuracy'].get(intent, 0)
            print(f'       {intent:<22}  F1={f1:.2%}  Recall={rec:.2%}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), f'{SAVE_DIR}/intent_classifier.pt')
            with open(f'{SAVE_DIR}/intent_classifier_metrics.json', 'w') as f:
                json.dump({
                    'epoch': epoch, 'val_accuracy': val_acc, 'macro_f1': macro_f1,
                    'train_accuracy': train_acc, 'loss': avg_loss, 'learning_rate': lr,
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
print(f'  Classifier done. Best val acc: {best_val_acc:.1%}')

# Final per-intent breakdown
classifier.load_state_dict(torch.load(f'{SAVE_DIR}/intent_classifier.pt', map_location='cpu'))
final = evaluate_classifier(classifier, val_data)
print(f'\n  Per-Intent Final (sorted by F1):')
print(f'  {"Intent":<22}  {"F1":>6}  {"Recall":>7}  {"N":>6}')
print(f'  {"-"*50}')
for intent, f1 in sorted(final['per_intent_f1'].items(), key=lambda x: x[1]):
    recall    = final['per_intent_accuracy'].get(intent, 0)
    n         = counts_after.get(intent, 0)
    indicator = '⚠' if f1 < 0.80 else '✓'
    print(f'  {indicator} {intent:<22}  {f1:>6.1%}  {recall:>7.1%}  {n:>6}')

# =========================================================
# STEP 2: Train Plan Scorer
# =========================================================
print(f'\n{separator()}')
print(f'  STEP 2 — Training Plan Scorer ({EPOCHS_SCORER} epochs)  [{now_ts()}]')
print(separator())

scorer        = PlanScoringNet(vocab_size=vocab_size, embed_dim=128, hidden_dim=128)
scr_optimizer = optim.AdamW(scorer.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
scr_scheduler = CosineAnnealingLR(scr_optimizer, T_max=EPOCHS_SCORER, eta_min=1e-6)
scr_loss_fn   = nn.BCELoss()

SCORER_DATA = [
    ('open app', 0.95), ('launch application', 0.94), ('close app', 0.93),
    ('web search', 0.92), ('search the internet', 0.91), ('web navigate', 0.91),
    ('browse to website', 0.90), ('web scrape', 0.88), ('file read', 0.91),
    ('file write', 0.88), ('file list', 0.92), ('memory store', 0.94),
    ('memory recall', 0.93), ('goal create', 0.91), ('goal list', 0.93),
    ('goal pause', 0.89), ('goal resume', 0.89), ('goal complete', 0.91),
    ('screen read', 0.87), ('active window', 0.88), ('list windows', 0.88),
    ('rollback', 0.91), ('confirm', 0.96), ('cancel', 0.96), ('reason', 0.82),
    ('identity query', 0.95), ('take screenshot', 0.90), ('system info', 0.88),
    ('generate report', 0.87), ('memory forget', 0.89), ('respond', 0.78),
    ('open app now', 0.94), ('please open that', 0.85), ('close that app', 0.90),
    ('search for something', 0.87), ('go to website', 0.89), ('read file', 0.90),
    ('write to file', 0.87), ('list directory', 0.91), ('remember this', 0.93),
    ('what do you remember', 0.92), ('create a goal', 0.90), ('mark done', 0.89),
    ('what is on screen', 0.86), ('capture screen', 0.89), ('check cpu', 0.87),
    ('do something', 0.60), ('help me', 0.62), ('i need something', 0.55),
    ('can you', 0.58), ('open it', 0.70), ('close that', 0.68),
    ('search for that thing', 0.65), ('go there', 0.63), ('do the task', 0.64),
    ('check the thing', 0.61), ('navigate somewhere', 0.65), ('write it down', 0.66),
    ('maybe open this', 0.55), ('try to find that', 0.58), ('show me stuff', 0.60),
    ('do something with it', 0.40), ('something about search', 0.38),
    ('maybe do that', 0.42), ('i think open', 0.40), ('possibly navigate', 0.43),
    ('sort of write', 0.38), ('kind of remember', 0.39),
    ('xyz123 nonsense', 0.15), ('invalid command', 0.20), ('blah blah', 0.10),
    ('random text', 0.12), ('', 0.05), ('unknown', 0.25), ('asdfgh', 0.18),
    ('delete all files', 0.25), ('format drive', 0.22), ('shutdown system', 0.32),
    ('kill process system', 0.35), ('rm -rf /', 0.18), ('delete system32', 0.12),
    ('wipe disk', 0.25), ('disable firewall', 0.30), ('remove all data', 0.23),
    ('destroy configuration', 0.26), ('erase everything', 0.20),
    ('factory reset', 0.28), ('overwrite boot sector', 0.15),
    ('terminate all processes', 0.35), ('kill all services', 0.32),
]

print(f'  Scorer training triples: {len(SCORER_DATA)}')
best_scr_loss = float('inf')

for epoch in range(1, EPOCHS_SCORER + 1):
    scorer.train()
    total_loss, num_samples = 0.0, 0
    random.shuffle(SCORER_DATA)

    for action_text, target_score in SCORER_DATA:
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

print(f'\n  ✓ Scorer done. Best loss: {best_scr_loss:.5f}')

# =========================================================
# FINAL SUMMARY
# =========================================================
print(f'\n{separator()}')
print(f'  AERIS FULL TRAINING COMPLETE  [{now_ts()}]')
print(separator())
print(f'  aeris_slm_best.pt          Already trained (val_ppl=2.4)')
print(f'  intent_classifier.pt       Val accuracy: {best_val_acc:.1%}')
print(f'  plan_scorer.pt             BCE loss: {best_scr_loss:.5f}')
print(f'  tokenizer.json             Vocab: {vocab_size:,} tokens')
print(separator())
print()
print('  AERIS is ready. Run: python main.py')
print(separator())
