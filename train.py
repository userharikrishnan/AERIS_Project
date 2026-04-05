"""
AERIS Production Training Script
Advanced training with stratified sampling, class balancing, and comprehensive validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

from models.slm import AerisSLM
from models.tokenizer import Tokenizer
from models.trainer import SLMTrainer
from models.intent_classifier import IntentClassifier, INTENT_TO_IDX, NUM_INTENT_CLASSES, IDX_TO_INTENT
from models.plan_scorer_model import PlanScoringNet

from data.aeris_training_data import TRAINING_PAIRS
from data.intent_training_data import INTENT_TRAINING_DATA

# -------------------------
# Production Config
# -------------------------
SEED = 42
EPOCHS_SLM = 50
EPOCHS_CLASSIFIER = 500
EPOCHS_SCORER = 300
LEARNING_RATE = 0.001
BATCH_SIZE = 64  # CRITICAL FIX 3: Increased from 32 to 64
PATIENCE = 150
VALIDATION_SPLIT = 0.2
MIN_SAMPLES_PER_CLASS = 100  # Minimum samples for stratification
SAVE_DIR = "checkpoints"

def set_seed(seed: int):
    """Reproducibility configuration"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# STEP 1: Enhanced Vocabulary Building
# =========================================================
print("=" * 70)
print("STEP 1: Building Production Vocabulary")
print("=" * 70)

tokenizer = Tokenizer(max_vocab_size=10000)

# Collect all text sources
all_texts = []
all_texts += [t for pair in TRAINING_PAIRS for t in pair]
all_texts += [text for text, _ in INTENT_TRAINING_DATA]

# CRITICAL FIX 1: ADD SPECIAL TOKENS FOR ROLE-AWARE TRAINING
special_tokens = ["<USER>", "<ASSISTANT>", "<EOS>"]
all_texts += special_tokens

# Domain-specific vocabulary for AERIS
domain_vocab = [
    # Applications
    "chrome", "firefox", "edge", "brave", "opera", "safari",
    "vscode", "sublime", "notepad", "notepad++", "vim", "emacs",
    "terminal", "cmd", "powershell", "gitbash", "anaconda",
    "spotify", "itunes", "vlc", "windows media player",
    "discord", "slack", "teams", "zoom", "skype", "webex",
    "steam", "epic games", "origin", "battle.net", "gog", "ubisoft connect",
    "outlook", "thunderbird", "gmail", "protonmail",
    "word", "excel", "powerpoint", "onenote", "libreoffice",
    "photoshop", "illustrator", "premiere", "after effects",
    "blender", "maya", "cinema 4d", "3ds max",
    "obs", "streamlabs", "xsplit",
    "postman", "insomnia", "docker desktop", "kubernetes",
    "mysql", "postgresql", "mongodb", "redis", "sqlite",
    "android studio", "intellij", "pycharm", "webstorm", "eclipse",
    "unity", "unreal engine", "godot", "game maker",
    
    # Web
    "google", "youtube", "github", "stackoverflow", "reddit",
    "twitter", "facebook", "linkedin", "instagram", "tiktok",
    "amazon", "ebay", "netflix", "spotify", "hulu", "disney plus",
    "wikipedia", "wikihow", "medium", "dev.to", "hashnode",
    "hackernews", "producthunt", "techcrunch", "the verge",
    "gmail", "outlook", "yahoo", "protonmail",
    "drive", "dropbox", "onedrive", "icloud", "google cloud",
    "notion", "trello", "asana", "monday", "jira", "confluence",
    "slack", "discord", "zoom", "webex", "teams",
    "udemy", "coursera", "edx", "khan academy", "skillshare",
    "leetcode", "hackerrank", "codewars", "exercism",
    "npm", "pypi", "maven", "gradle", "cargo", "nuget",
    "dockerhub", "kubernetes", "terraform", "ansible",
    "aws", "azure", "gcp", "digitalocean", "linode", "vultr",
    "localhost", "127.0.0.1", "0.0.0.0", "port",
    
    # File operations
    "config", "configuration", "settings", "preferences", "options",
    "json", "xml", "yaml", "yml", "toml", "ini", "conf",
    "txt", "text", "md", "markdown", "rst",
    "pdf", "doc", "docx", "odt", "rtf",
    "csv", "tsv", "xlsx", "xls", "ods",
    "py", "python", "js", "javascript", "ts", "typescript",
    "html", "css", "scss", "sass", "less",
    "java", "cpp", "c++", "c", "csharp", "cs", "go", "rust", "ruby", "php",
    "sql", "sh", "bash", "zsh", "fish", "ps1",
    "log", "logs", "debug", "trace", "error",
    "backup", "bak", "old", "archive", "zip", "tar", "gz", "rar", "7z",
    "readme", "license", "changelog", "todo", "contributing",
    "gitignore", "dockerignore", "eslintignore",
    
    # System
    "screenshot", "screen capture", "clipboard", "copy", "paste",
    "process", "task", "thread", "service", "daemon",
    "window", "dialog", "popup", "notification", "toast",
    "desktop", "taskbar", "start menu", "system tray",
    "folder", "directory", "path", "file", "filename", "extension",
    "root", "home", "user", "admin", "system", "local", "remote",
    
    # Intent keywords
    "launch", "open", "start", "run", "execute", "initiate", "begin",
    "close", "exit", "quit", "terminate", "kill", "stop", "end",
    "search", "find", "look up", "google", "query", "lookup",
    "navigate", "browse", "visit", "go to", "access", "load",
    "create", "make", "new", "generate", "produce", "build",
    "read", "view", "show", "display", "open", "cat", "type",
    "write", "save", "store", "export", "output", "write to",
    "delete", "remove", "erase", "trash", "destroy", "wipe",
    "list", "show", "display", "enumerate", "catalog", "ls", "dir",
    "remember", "memorize", "store", "save", "keep", "retain",
    "recall", "retrieve", "fetch", "get", "bring up", "access",
    "forget", "delete", "clear", "erase", "wipe", "remove",
    "goal", "objective", "target", "aim", "purpose", "mission",
    "plan", "strategy", "approach", "method", "way",
    "pause", "suspend", "hold", "freeze", "halt", "stop",
    "resume", "continue", "proceed", "unpause", "restart",
    "complete", "finish", "done", "accomplish", "achieve",
    "screen", "display", "monitor", "vision", "see", "look", "view",
    "window", "application", "app", "program", "process",
    "undo", "rollback", "revert", "reverse", "go back", "restore",
    "confirm", "approve", "accept", "yes", "proceed", "go ahead",
    "cancel", "abort", "stop", "no", "reject", "decline",
    "reason", "think", "analyze", "evaluate", "assess", "consider",
    "identity", "who are you", "what are you", "name", "purpose",
    "chat", "talk", "conversation", "discuss", "speak", "communicate",
    
    # Emotional/social
    "happy", "sad", "angry", "frustrated", "excited", "bored",
    "tired", "stressed", "worried", "scared", "confused", "help",
    "thanks", "thank you", "please", "sorry", "excuse me",
    "good morning", "good afternoon", "good evening", "good night",
    "hello", "hi", "hey", "greetings", "howdy", "what's up",
    "goodbye", "bye", "see you", "later", "take care",
    
    # Status
    "status", "health", "state", "condition", "report", "diagnostics",
    "online", "offline", "active", "inactive", "ready", "busy",
    "error", "warning", "alert", "critical", "emergency", "fatal",
    "success", "completed", "failed", "pending", "processing",
]

all_texts += domain_vocab
tokenizer.train(all_texts, min_freq=1)

vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size} words")
print(f"Domain terms added: {len(domain_vocab)}")
print(f"Special tokens added: {special_tokens}")

# Save tokenizer
tokenizer.save(f"{SAVE_DIR}/tokenizer.json")

# =========================================================
# PRE-TOKENIZE SLM DATA (CRITICAL SPEED FIX)
# =========================================================
print("\nPre-tokenizing SLM dataset...")

TOKENIZED_PAIRS = []

for input_text, response in TRAINING_PAIRS:
    input_ids = tokenizer.encode("<USER> " + input_text + " <ASSISTANT>")
    target_ids = tokenizer.encode(response + " <EOS>")
    
    if len(input_ids) > 0 and len(target_ids) > 1:
        TOKENIZED_PAIRS.append((input_ids, target_ids))

print(f"Tokenized {len(TOKENIZED_PAIRS)} samples")

# =========================================================
# STEP 2: Train SLM (Response Generation) — PERFECT NEXT-TOKEN LEARNING
# =========================================================
print("\n" + "=" * 70)
print(f"STEP 2: Training SLM — {EPOCHS_SLM} epochs")
print("=" * 70)

slm = AerisSLM(vocab_size=vocab_size, embed_dim=128, hidden_dim=256)
trainer = SLMTrainer(slm, lr=LEARNING_RATE)
slm_scheduler = CosineAnnealingLR(trainer.optimizer, T_max=EPOCHS_SLM, eta_min=1e-6)

# CRITICAL FIX: PERFECT make_slm_batch with proper next-token shift
def make_slm_batch(pairs, batch_size=8):
    """Create batched training data with role-aware formatting and proper next-token prediction"""
    batch_inputs, batch_targets = [], []

    for input_ids, target_ids in pairs:
        # PERFECT FIX: Proper next-token learning with shift
        # Input: all tokens (user + assistant prompt + response without last token)
        # Target: shift by 1 (pads for user portion + response tokens shifted)
        full_input = input_ids + target_ids[:-1]
        full_target = ([0] * len(input_ids)) + target_ids[1:]

        # Lengths now perfectly aligned
        if len(full_input) < 2:
            continue

        batch_inputs.append(full_input)
        batch_targets.append(full_target)

        if len(batch_inputs) == batch_size:
            max_len = max(len(x) for x in batch_inputs)

            padded_inputs = [x + [0] * (max_len - len(x)) for x in batch_inputs]
            padded_targets = [x + [0] * (max_len - len(x)) for x in batch_targets]

            yield torch.tensor(padded_inputs), torch.tensor(padded_targets)

            batch_inputs, batch_targets = [], []

    if batch_inputs:
        max_len = max(len(x) for x in batch_inputs)

        padded_inputs = [x + [0] * (max_len - len(x)) for x in batch_inputs]
        padded_targets = [x + [0] * (max_len - len(x)) for x in batch_targets]

        yield torch.tensor(padded_inputs), torch.tensor(padded_targets)

# CRITICAL FIX 5: Mixed precision setup
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# ── STEP 2 training loop (improved with perplexity tracking) ──
best_slm_loss    = float("inf")
patience_counter = 0

for epoch in range(1, EPOCHS_SLM + 1):
    # CRITICAL FIX 2: Shuffle training data before each epoch
    shuffled_pairs = TOKENIZED_PAIRS.copy()
    random.shuffle(shuffled_pairs)
    
    total_loss   = 0.0
    total_ppl    = 0.0
    num_batches  = 0

    for batch_idx, (inputs, targets) in enumerate(make_slm_batch(shuffled_pairs, batch_size=BATCH_SIZE)):
        
        if batch_idx % 50 == 0:
            print(f"    Batch {batch_idx} running...")
        
        # CRITICAL FIX 5: Mixed precision training
        if scaler and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                result = trainer.train_step(inputs, targets)
        else:
            result = trainer.train_step(inputs, targets)
            
        total_loss  += result['loss']
        total_ppl   += result['perplexity']
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_ppl  = total_ppl  / num_batches if num_batches > 0 else 0
    slm_scheduler.step()

    # Early stopping check
    if avg_loss < best_slm_loss:
        best_slm_loss = avg_loss
        torch.save(slm.state_dict(), f"{SAVE_DIR}/aeris_slm_best.pt")
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 100 == 0 or epoch == 1:
        lr = slm_scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:>5}/{EPOCHS_SLM} | Loss: {avg_loss:.4f} | PPL: {avg_ppl:.1f} | LR: {lr:.6f} | Patience: {patience_counter}")

    if patience_counter >= PATIENCE:
        print(f"  ✓ Early stopping at epoch {epoch}")
        break

torch.save(slm.state_dict(), f"{SAVE_DIR}/aeris_slm.pt")
print(f"\n✓ Best SLM loss: {best_slm_loss:.4f}")

# =========================================================
# STEP 3: Train Intent Classifier — IMPROVED VERSION
# =========================================================
print("\n" + "=" * 70)
print(f"STEP 3: Training Intent Classifier — {EPOCHS_CLASSIFIER} epochs")
print("=" * 70)

# ------------------------------------------------------------------
# 3a. Augment UNKNOWN class (zero samples → ~200 OOD examples)
# ------------------------------------------------------------------
OOD_TEXTS = [
    "asdfghjkl", "qqqqqq", "12345678", "!@#$%^", "zzzzz",
    "xkcd random", "fjfjfj", "lorem ipsum dolor", "foo bar baz",
    "test test test", "aabbcc", "random gibberish here",
    "zxcvbnm", "undefined behaviour", "null pointer exception",
    "stack overflow error", "segfault core dump", "divide by zero",
    "syntax error on line", "unresolved import", "module not found",
    "connection reset by peer", "timeout expired", "503 service unavailable",
    "what is the meaning of life", "tell me a story about dragons",
    "explain quantum entanglement", "what is consciousness",
    "i like turtles very much", "bananas are yellow sometimes",
    "the quick brown fox jumps", "supercalifragilistic",
    "pneumonoultramicroscopicsilicovolcanoconiosis",
] * 6   # ~200 examples

INTENT_TRAINING_DATA_AUG = list(INTENT_TRAINING_DATA)
for text in OOD_TEXTS:
    INTENT_TRAINING_DATA_AUG.append((text, "UNKNOWN"))

# ------------------------------------------------------------------
# 3b. Compute class weights (inverse frequency, capped at 10x)
# ------------------------------------------------------------------
intent_counts = Counter([intent for _, intent in INTENT_TRAINING_DATA_AUG])
total_samples = sum(intent_counts.values())
num_classes   = len(INTENT_TO_IDX)

# CRITICAL FIX 6: Print intent distribution
print("\nIntent Distribution (before balancing):")
for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
    print(f"  {intent:22s}: {count:5d}")

# Weight = total / (num_classes * class_count), capped
class_weights = torch.zeros(num_classes)
for intent, idx in INTENT_TO_IDX.items():
    count = intent_counts.get(intent, 1)
    weight = total_samples / (num_classes * count)
    class_weights[idx] = min(weight, 10.0)   # cap at 10× to prevent instability

print(f"\nClass weights (top 5 most weighted):")
top_weights = sorted(
    [(w.item(), intent) for intent, w in zip(INTENT_TO_IDX.keys(), class_weights)],
    reverse=True
)[:5]
for w, intent in top_weights:
    print(f"  {intent:22s}: {w:.2f}×")

# ------------------------------------------------------------------
# 3c. Oversample minority classes to MIN_SAMPLES_PER_CLASS
# ------------------------------------------------------------------
MIN_SAMPLES_PER_CLASS = 250

augmented = list(INTENT_TRAINING_DATA_AUG)
by_intent = defaultdict(list)
for text, intent in augmented:
        by_intent[intent].append((text, intent))

oversampled = list(augmented)
for intent, samples in by_intent.items():
    if len(samples) < MIN_SAMPLES_PER_CLASS:
        deficit = MIN_SAMPLES_PER_CLASS - len(samples)
        extras  = random.choices(samples, k=deficit)
        oversampled.extend(extras)

# Extra boost for CHAT class
chat_samples = [s for s in oversampled if s[1] == "CHAT"]

if len(chat_samples) > 0:
    oversampled.extend(random.choices(chat_samples, k=len(chat_samples) * 3))
    print(f"\n✓ CHAT class boosted: added {len(chat_samples) * 3} extra samples")

random.shuffle(oversampled)

print(f"\nAfter oversampling:")
counts_after = Counter([i for _, i in oversampled])
for intent, count in sorted(counts_after.items(), key=lambda x: -x[1]):
    pct = count / len(oversampled) * 100
    bar = "█" * int(pct / 2)
    print(f"  {intent:22s}: {count:5d} ({pct:4.1f}%) {bar}")

# Hard negative training examples
HARD_NEGATIVES = [
    ("can you open chrome", "OPEN_APP"),
    ("can you tell me something", "CHAT"),
    ("can you explain recursion", "CHAT"),
    ("can you open something", "OPEN_APP"),
]

oversampled.extend(HARD_NEGATIVES)
print(f"\n✓ Added {len(HARD_NEGATIVES)} hard negative examples")

# Check for label mismatches (now includes UNKNOWN)
training_intents = set(counts_after.keys())
model_intents    = set(INTENT_TO_IDX.keys())
missing_in_model = training_intents - model_intents
if missing_in_model:
    print(f"\n❌ CRITICAL: Intents in training but not in model: {missing_in_model}")
    raise ValueError("Intent label mismatch — fix INTENT_TO_IDX")

# ------------------------------------------------------------------
# 3d. Stratified split (CRITICAL FIX 8: Proper validation split)
# ------------------------------------------------------------------
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
print(f"\nSplit: Train={len(train_data)} | Val={len(val_data)}")

# ------------------------------------------------------------------
# 3e. Model, optimizer, weighted loss
# ------------------------------------------------------------------
classifier = IntentClassifier(
    vocab_size=vocab_size,
    embed_dim=128,
    hidden_dim=256,
    num_heads=4,
    dropout=0.3,
)

clf_optimizer = optim.AdamW(
    classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)
clf_scheduler = CosineAnnealingLR(
    clf_optimizer,
    T_max=EPOCHS_CLASSIFIER,
    eta_min=1e-6,
)

# Weighted CE + label smoothing
clf_loss_fn = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1,
)

# ------------------------------------------------------------------
# 3f. Batch generator (stratified, balanced)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 3g. Evaluation helper
# ------------------------------------------------------------------
def evaluate_classifier(model, data):
    model.eval()
    correct, total = 0, 0
    intent_correct = defaultdict(int)
    intent_total   = defaultdict(int)

    with torch.no_grad():
        for text, intent_label in data:
            ids = tokenizer.encode(text.lower())
            if not ids:
                continue
            inputs = torch.tensor([ids])
            result = model.predict_with_confidence(inputs)
            predicted = result['intent']
            intent_total[intent_label]   += 1
            if predicted == intent_label:
                correct += 1
                intent_correct[intent_label] += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    per_intent = {
        k: intent_correct[k] / intent_total[k]
        if intent_total[k] > 0 else 0.0
        for k in intent_total
    }
    return {'accuracy': accuracy, 'per_intent_accuracy': per_intent, 'total': total}

# ------------------------------------------------------------------
# 3h. Training loop
# ------------------------------------------------------------------
best_val_acc    = 0.0
patience_counter = 0

for epoch in range(1, EPOCHS_CLASSIFIER + 1):
    classifier.train()
    total_loss, num_batches = 0.0, 0

    for batch in make_clf_batch(train_data):
        texts, labels = zip(*batch)

        encoded = [tokenizer.encode(t.lower()) for t in texts]
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

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    clf_scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        train_m = evaluate_classifier(classifier, random.sample(train_data, min(500, len(train_data))))
        val_m   = evaluate_classifier(classifier, val_data)
        train_acc, val_acc = train_m['accuracy'], val_m['accuracy']
        lr = clf_scheduler.get_last_lr()[0]

        print(f"\n  Epoch {epoch:>4}/{EPOCHS_CLASSIFIER}")
        print(f"    Loss: {avg_loss:.4f} | LR: {lr:.6f}")
        print(f"    Train Acc: {train_acc:.1%} | Val Acc: {val_acc:.1%}")

        # Print worst-performing intents
        worst = sorted(val_m['per_intent_accuracy'].items(), key=lambda x: x[1])[:5]
        print(f"    Worst 5 intents:")
        for intent, acc in worst:
            print(f"      {intent:22s}: {acc:.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), f"{SAVE_DIR}/intent_classifier.pt")
            with open(f"{SAVE_DIR}/intent_classifier_metrics.json", "w") as f:
                json.dump({
                    'epoch': epoch,
                    'val_accuracy': val_acc,
                    'train_accuracy': train_acc,
                    'loss': avg_loss,
                    'learning_rate': lr,
                    'per_intent_accuracy': val_m['per_intent_accuracy'],
                }, f, indent=2)
            patience_counter = 0
            print(f"    ✓ New best: {val_acc:.1%}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE // 2:
            print(f"    ✓ Early stopping at epoch {epoch}")
            break

print(f"\n{'='*70}")
print(f"Best Validation Accuracy: {best_val_acc:.1%}")

# Final detailed breakdown
classifier.load_state_dict(torch.load(f"{SAVE_DIR}/intent_classifier.pt"))
final = evaluate_classifier(classifier, val_data)
print(f"\nPer-Intent Validation Accuracy (all):")
for intent, acc in sorted(final['per_intent_accuracy'].items(), key=lambda x: x[1]):
    count = counts_after.get(intent, 0)
    if count > 0:
        print(f"  {intent:22s}: {acc:.1%}  (n={count})")

# =========================================================
# STEP 4: Train Plan Scorer with Enhanced Data
# =========================================================
print("\n" + "=" * 70)
print(f"STEP 4: Training Plan Scorer — {EPOCHS_SCORER} epochs")
print("=" * 70)

scorer = PlanScoringNet(vocab_size=vocab_size, embed_dim=128, hidden_dim=128)
scr_optimizer = optim.AdamW(scorer.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
scr_scheduler = CosineAnnealingLR(scr_optimizer, T_max=EPOCHS_SCORER, eta_min=1e-6)
scr_loss_fn = nn.BCELoss()

# Enhanced training data with risk calibration
SCORER_TRAINING_DATA = [
    # High confidence positive examples (routine operations)
    ("open app", 0.95), ("close app", 0.93),
    ("web search", 0.92), ("web navigate", 0.90),
    ("file read", 0.91), ("file write", 0.88),
    ("file list", 0.92), ("memory store", 0.94),
    ("memory recall", 0.93), ("goal create", 0.90),
    ("goal list", 0.93), ("goal pause", 0.89),
    ("goal resume", 0.89), ("goal complete", 0.90),
    ("screen read", 0.87), ("active window", 0.88),
    ("list windows", 0.88), ("rollback", 0.91),
    ("confirm", 0.96), ("cancel", 0.96),
    ("reason", 0.82), ("identity query", 0.95),
    ("respond", 0.78),
    
    # Medium confidence (ambiguous or context-dependent)
    ("do something", 0.60), ("help me", 0.65),
    ("i need something", 0.55), ("can you", 0.62),
    ("open it", 0.70), ("close that", 0.68),
    ("search for that thing", 0.65), ("go there", 0.63),
    
    # Low confidence negative examples (nonsense/invalid)
    ("xyz123 nonsense", 0.15), ("invalid command", 0.20),
    ("blah blah", 0.10), ("random text", 0.12),
    ("", 0.05), ("unknown", 0.25), ("asdfgh", 0.18),
    
    # Dangerous operations (should trigger verification)
    ("delete all files", 0.30), ("format drive", 0.25),
    ("shutdown system", 0.35), ("kill process system", 0.40),
    ("rm -rf /", 0.20), ("delete system32", 0.15),
    ("wipe disk", 0.25), ("disable firewall", 0.35),
    
    # Unclear/ambiguous
    ("maybe do something", 0.45), ("possibly search", 0.42),
    ("i guess open", 0.40), ("perhaps navigate", 0.43),
]

best_scr_loss = float("inf")

for epoch in range(1, EPOCHS_SCORER + 1):
    scorer.train()
    total_loss = 0.0
    num_samples = 0
    
    # Shuffle data each epoch
    random.shuffle(SCORER_TRAINING_DATA)
    
    for action_text, target_score in SCORER_TRAINING_DATA:
        token_ids = tokenizer.encode(action_text)
        if not token_ids:
            continue
        
        input_t = torch.tensor([token_ids])
        target = torch.tensor([[target_score]])
        
        scr_optimizer.zero_grad()
        pred = scorer(input_t)
        loss = scr_loss_fn(pred, target)
        loss.backward()
        scr_optimizer.step()
        
        total_loss += loss.item()
        num_samples += 1
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    scr_scheduler.step()
    
    if avg_loss < best_scr_loss:
        best_scr_loss = avg_loss
        torch.save(scorer.state_dict(), f"{SAVE_DIR}/plan_scorer.pt")
    
    if epoch % 50 == 0 or epoch == 1:
        lr = scr_scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:>4}/{EPOCHS_SCORER} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

print(f"\n✓ Best scorer loss: {best_scr_loss:.4f}")

# =========================================================
# Summary
# =========================================================
print("\n" + "=" * 70)
print("PRODUCTION TRAINING COMPLETE")
print("=" * 70)
print(f"  aeris_slm.pt           Response generation (loss: {best_slm_loss:.4f})")
print(f"  aeris_slm_best.pt      Best SLM checkpoint")
print(f"  intent_classifier.pt   Intent classification (Val: {best_val_acc:.1%})")
print(f"  intent_classifier_metrics.json  Detailed per-intent metrics")
print(f"  plan_scorer.pt         Confidence scoring (loss: {best_scr_loss:.4f})")
print(f"  tokenizer.json         Vocabulary ({vocab_size} words)")
print("=" * 70)