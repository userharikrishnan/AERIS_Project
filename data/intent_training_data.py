"""
AERIS intent training data — upgraded version.

Goals:
- strengthen weak semantic classes
- reduce CHAT/command confusion
- add hard negatives and overlap traps
- keep the classifier fully custom and self-trained
"""

from __future__ import annotations

import random
from collections import Counter
from typing import List, Tuple

random.seed(42)

INTENT_TRAINING_DATA: List[Tuple[str, str]] = []

def add(text: str, label: str) -> None:
    text = " ".join(text.strip().split())
    if text:
        INTENT_TRAINING_DATA.append((text, label))

def add_many(label: str, items) -> None:
    for item in items:
        add(item, label)

def expand(label: str, templates, topics) -> None:
    for topic in topics:
        for tpl in templates:
            add(tpl.format(x=topic), label)

def dedupe_pairs(pairs):
    seen = set()
    out = []
    for text, label in pairs:
        key = (text.lower().strip(), label)
        if key not in seen:
            seen.add(key)
            out.append((text, label))
    return out

APP_NAMES = [
    "chrome", "firefox", "edge", "safari", "opera", "brave",
    "vscode", "sublime", "notepad", "notepad++", "vim", "emacs",
    "spotify", "itunes", "vlc", "windows media player",
    "discord", "slack", "teams", "zoom", "skype", "webex",
    "steam", "epic games", "origin", "battle.net",
    "outlook", "thunderbird", "gmail",
    "word", "excel", "powerpoint", "libreoffice",
    "calculator", "calendar", "camera", "photos", "settings",
    "task manager", "control panel", "file explorer", "terminal",
    "cmd", "powershell", "git bash", "anaconda prompt",
    "photoshop", "illustrator", "premiere", "after effects",
    "blender", "maya", "cinema 4d",
    "obs", "streamlabs", "xsplit",
    "postman", "insomnia", "docker desktop", "kubernetes",
    "mysql workbench", "pgadmin", "dbeaver",
    "android studio", "intellij", "pycharm", "webstorm",
    "unity", "unreal engine", "godot",
    "epic games launcher", "gog galaxy", "ubisoft connect",
    "c drive", "d drive", "e drive", "f drive",
    "the c drive", "my c drive", "c partition", "local disk c",
    "downloads folder", "documents folder", "desktop", "my pictures",
    "the directory", "the folder", "system drive", "users folder",
    "music folder", "videos folder", "appdata", "program files",
    "windows explorer", "explorer", "my computer", "this pc"
]

WEBSITES = [
    "google.com", "youtube.com", "github.com", "stackoverflow.com",
    "reddit.com", "twitter.com", "facebook.com", "linkedin.com",
    "amazon.com", "ebay.com", "netflix.com", "spotify.com",
    "wikipedia.org", "wikihow.com", "medium.com", "dev.to",
    "news.ycombinator.com", "producthunt.com", "techcrunch.com",
    "theverge.com", "arstechnica.com", "github.com/trending",
    "gmail.com", "outlook.com", "yahoo.com", "protonmail.com",
    "calendar.google.com", "drive.google.com", "docs.google.com",
    "notion.so", "trello.com", "asana.com", "monday.com",
    "slack.com", "discord.com", "zoom.us", "webex.com",
    "udemy.com", "coursera.org", "edx.org", "khanacademy.org",
    "leetcode.com", "hackerrank.com", "codewars.com",
    "npmjs.com", "pypi.org", "maven.apache.org", "gradle.org",
    "dockerhub.com", "kubernetes.io", "terraform.io",
    "aws.amazon.com", "azure.microsoft.com", "cloud.google.com",
    "heroku.com", "vercel.com", "netlify.com", "digitalocean.com",
    "localhost:3000", "localhost:8080", "127.0.0.1:5000",
    "my dashboard", "the admin panel", "settings page",
    "login page", "signup page", "profile page",
    "home page", "main page", "landing page",
    "documentation", "api docs", "help center",
    "support page", "contact page", "about page",
]

FILES = [
    "document.txt", "report.pdf", "data.csv", "script.py", "code.js",
    "readme.md", "config.json", "settings.xml", "log.txt", "backup.zip",
    "image.png", "photo.jpg", "video.mp4", "audio.mp3", "archive.tar",
    "database.db", "spreadsheet.xlsx", "presentation.pptx", "notes.docx",
    "main.py", "app.py", "server.js", "index.html", "style.css",
    "requirements.txt", "package.json", "dockerfile", "Makefile",
    ".gitignore", "LICENSE", "README", "CHANGELOG", "TODO",
    "my file", "the document", "that file", "this file",
    "project files", "source code", "configuration", "logs",
    "downloads", "documents", "pictures", "music", "videos",
]

PATHS = [
    "desktop", "downloads", "documents", "pictures", "home",
    "root", "current folder", "this directory", "project folder",
    "c drive", "d drive", "external drive", "usb", "cloud",
    "google drive", "dropbox", "one drive", "icloud",
    "github repo", "local repository", "workspace",
]

MEMORY_TOPICS = [
    "my name", "my birthday", "my address", "my phone number",
    "my email", "my password", "my preferences", "my settings",
    "the meeting", "the deadline", "the appointment", "the event",
    "the task", "the project", "the goal", "the objective",
    "the idea", "the thought", "the concept", "the plan",
    "the conversation", "what we discussed", "our talk",
    "the note", "the reminder", "the alarm", "the schedule",
    "the list", "the items", "the inventory", "the catalog",
    "the password for {x}", "the login for {x}", "the account {x}",
    "the wifi password", "the door code", "the pin",
    "my favorite {x}", "my preference for {x}", "my setting for {x}",
    "the important thing", "the key point", "the main idea",
    "this for later", "that for future reference", "this information",
]

# CHAT
add_many("CHAT", [
    "hello there", "hey aeris", "what's up", "can we talk", "tell me something",
    "how are you", "i need help", "thanks", "thank you", "that's funny",
    "you are useful", "this is confusing", "explain again", "be brief",
    "be detailed", "that was helpful", "i am annoyed", "i am happy",
    "i am in a hurry", "just answer plainly", "good morning", "good night",
    "nice work", "you're fast", "that is awesome", "cool", "okay then",
    "hmm", "interesting", "all right", "let's go", "yo",
    "bro", "mate", "got it", "perfect", "sounds good"
])

# IDENTITY / CONFIRM / CANCEL / REASONING
add_many("IDENTITY_QUERY", [
    "who are you", "what are you", "what is your name", "what can you do",
    "are you ai", "are you human", "are you conscious", "what is your purpose",
    "how do you work", "what makes you different", "tell me about yourself",
    "what is aeris", "are you a robot", "are you alive"
])
add_many("CONFIRM", [
    "yes", "yeah", "yep", "okay", "ok", "confirm", "proceed", "go ahead",
    "continue", "do it", "please do", "sounds good", "that is fine", "sure",
    "absolutely", "affirmative"
])
add_many("CANCEL", [
    "no", "nope", "stop", "cancel", "abort", "never mind", "forget it",
    "don't do it", "pause", "hold on", "wait", "stop that", "ignore that"
])
add_many("REASONING", [
    "should i do this", "what is the best option", "compare the choices",
    "what should i choose", "which is safer", "which is faster",
    "which is cheaper", "how do i decide", "what is the tradeoff",
    "help me reason about this", "what would you recommend",
    "what is the better path", "what is the smartest move"
])

# MEMORY
for topic in MEMORY_TOPICS:
    t = topic.format(x="something")
    if topic == "my name":
        add("remember that my project name is aeris", "MEMORY_STORE")
    add(f"remember {t if '{x}' in topic else topic}", "MEMORY_STORE")
    add(f"save {t if '{x}' in topic else topic}", "MEMORY_STORE")
    add(f"store {t if '{x}' in topic else topic}", "MEMORY_STORE")
    add(f"keep {t if '{x}' in topic else topic}", "MEMORY_STORE")
    add(f"note {t if '{x}' in topic else topic}", "MEMORY_STORE")

for text in [
    "what do you remember", "recall my preferences", "do you remember what i said",
    "what did i tell you earlier", "bring up my note", "what is stored",
    "fetch the memory", "tell me what you saved", "what have you stored about me",
    "what do you know about my preferences", "do you remember my last instruction"
]:
    add(text, "MEMORY_RECALL")

add_many("MEMORY_FORGET", [
    "forget that", "delete that memory", "erase the note", "clear that",
    "remove it from memory", "forget my previous instruction",
    "forget the last thing", "erase my previous request"
])

# GOALS
for text in [
    "create a goal", "set a goal", "start a goal", "list my goals",
    "show my goals", "pause the current goal", "resume the goal",
    "complete the goal", "mark the goal done", "cancel the goal",
    "what goals are active", "what is my current goal", "track this task",
    "pause task tracking", "resume task tracking", "finish the objective"
]:
    if "list" in text or "show" in text or "what goals" in text:
        add(text, "GOAL_LIST")
    elif "pause" in text:
        add(text, "GOAL_PAUSE")
    elif "resume" in text:
        add(text, "GOAL_RESUME")
    elif "complete" in text or "done" in text or "finish" in text:
        add(text, "GOAL_COMPLETE")
    elif "cancel" in text:
        add(text, "CANCEL")
    else:
        add(text, "GOAL_CREATE")

# SCREEN/VISION/SYSTEM
for text in [
    "what is on my screen", "read the screen", "take a screenshot", "list open windows",
    "which window is active", "what app is focused", "show system info",
    "what system information can you see", "inspect the display", "what is visible",
    "what do you see", "analyze the screen", "check the current window"
]:
    if "screenshot" in text:
        add(text, "SCREENSHOT")
    elif "window is active" in text or "app is focused" in text or "current window" in text:
        add(text, "ACTIVE_WINDOW")
    elif "open windows" in text or "list windows" in text:
        add(text, "LIST_WINDOWS")
    elif "system info" in text or "system information" in text:
        add(text, "SYSTEM_INFO")
    elif "see" in text or "visible" in text:
        add(text, "VISION_QUERY")
    else:
        add(text, "READ_SCREEN")

# OPEN/CLOSE APP
OPEN_TEMPLATES = [
    "open {x}", "launch {x}", "start {x}", "run {x}", "begin {x}",
    "can you open {x}", "please open {x}", "would you open {x}",
    "fire up {x}", "bring up {x}", "get {x} going", "open {x} now",
    "open {x} for me", "start {x} now", "open the {x}", "launch the {x}"
]
CLOSE_TEMPLATES = [
    "close {x}", "exit {x}", "quit {x}", "terminate {x}", "shut down {x}",
    "can you close {x}", "please close {x}", "would you exit {x}",
    "kill {x}", "stop {x}", "close the {x}", "exit the {x}"
]
expand("OPEN_APP", OPEN_TEMPLATES, APP_NAMES)
expand("CLOSE_APP", CLOSE_TEMPLATES, APP_NAMES)
add_many("OPEN_APP", [
    "open the browser", "launch the browser", "open chrome", "start vscode",
    "bring up chrome", "open the app", "open the editor", "open the terminal",
    "switch to chrome", "focus chrome", "open the music app"
])
add_many("CLOSE_APP", [
    "close the browser", "close chrome", "quit chrome", "close the app",
    "close the current window", "exit the program", "kill this process",
    "shut this down", "close everything"
])

# WEB
SEARCH_TOPICS = [
    "python tutorials", "machine learning", "deep learning", "neural networks",
    "artificial intelligence", "data science", "web development",
    "javascript frameworks", "react vs vue", "docker tutorial",
    "kubernetes basics", "cloud computing", "aws vs azure",
    "best programming languages", "how to code", "learn coding",
    "stock prices today", "weather forecast", "news today",
    "latest technology", "space exploration", "mars mission",
    "climate change", "renewable energy", "electric vehicles",
    "crypto prices", "bitcoin", "ethereum", "blockchain",
    "recipes for dinner", "healthy meals", "quick recipes",
    "workout routines", "fitness tips", "yoga for beginners",
    "meditation techniques", "mental health", "productivity tips",
    "time management", "project management", "agile methodology",
    "best laptops 2024", "smartphone reviews", "tech news",
    "gaming laptops", "pc building guide", "gpu prices",
    "travel destinations", "cheap flights", "hotel booking",
    "restaurant near me", "movie reviews", "tv shows to watch",
    "book recommendations", "music playlists", "podcast recommendations",
    "how to invest", "personal finance", "budgeting tips",
    "tax information", "insurance guide", "mortgage calculator",
    "medical symptoms", "health information", "doctor appointment",
    "legal advice", "government services", "tax filing",
    "university courses", "online learning", "certification programs",
    "job openings", "resume tips", "interview preparation",
    "salary negotiation", "career advice", "freelancing tips",
]
SEARCH_TEMPLATES = [
    "search for {x}", "search {x}", "google {x}", "look up {x}",
    "find {x}", "look for {x}", "search up {x}", "web search for {x}",
    "search the web for {x}", "google search for {x}", "internet search for {x}",
    "find information about {x}", "look up information on {x}",
    "research {x}", "find details about {x}", "get information on {x}",
    "search online for {x}", "browse for {x}", "query {x}",
    "can you search for {x}", "please search {x}", "could you look up {x}",
    "i need to find {x}", "help me search for {x}", "i want to know about {x}",
]
for topic in SEARCH_TOPICS:
    expand("WEB_SEARCH", SEARCH_TEMPLATES, [topic])
NAVIGATE_TEMPLATES = [
    "go to {x}", "navigate to {x}", "open {x}", "visit {x}",
    "browse to {x}", "access {x}", "load {x}", "enter {x}",
    "take me to {x}", "bring me to {x}", "send me to {x}",
    "redirect to {x}", "switch to {x}", "change to {x}",
    "jump to {x}", "move to {x}", "could you open {x}",
    "i need to access {x}", "take me to {x} please",
    "get me to {x}", "bring up {x}", "show me {x}",
]
for site in WEBSITES:
    expand("WEB_NAVIGATE", NAVIGATE_TEMPLATES, [site])
add_many("WEB_SEARCH", [
    "search", "search the internet", "web search", "google it",
    "look it up", "search again", "refine search", "find it online",
    "i need to google something", "can you google that"
])
add_many("WEB_NAVIGATE", [
    "go to google.com", "open github.com", "visit stackoverflow.com",
    "take me to gmail", "go to the dashboard", "open youtube.com"
])

# FILES
READ_TEMPLATES = [
    "read {x}", "open {x}", "show {x}", "display {x}", "view {x}",
    "print {x}", "output {x}", "cat {x}", "type {x}", "echo {x}",
    "contents of {x}", "inside {x}", "what is in {x}", "show me {x}",
    "display contents of {x}", "open file {x}", "read file {x}",
    "what does {x} say", "what is written in {x}", "text of {x}",
    "content of {x}", "parse {x}", "scan {x}", "inspect {x}", "review {x}"
]
WRITE_TEMPLATES = [
    "create {x}", "write {x}", "make {x}", "generate {x}", "produce {x}",
    "save to {x}", "write to {x}", "export to {x}", "create file {x}",
    "save as {x}", "write data to {x}", "edit {x}", "modify {x}", "update {x}"
]
DELETE_TEMPLATES = [
    "delete {x}", "remove {x}", "erase {x}", "destroy {x}", "eliminate {x}",
    "trash {x}", "rm {x}", "wipe {x}", "purge {x}", "discard {x}",
    "secure erase {x}", "unlink {x}", "delete folder {x}", "remove folder {x}"
]
LIST_TEMPLATES = [
    "list {x}", "show {x}", "display {x}", "view {x}", "see {x}",
    "dir {x}", "ls {x}", "directory {x}", "folder contents {x}",
    "what is in {x}", "contents of {x}", "files in {x}", "tree view",
    "directory listing", "folder structure", "show files", "list files"
]
expand("FILE_READ", READ_TEMPLATES, FILES)
expand("FILE_WRITE", WRITE_TEMPLATES, FILES)
expand("FILE_DELETE", DELETE_TEMPLATES, FILES[:30])
expand("FILE_LIST", LIST_TEMPLATES, PATHS)
add_many("FILE_READ", ["read file.txt", "show report.pdf", "open config.json", "display the log file"])
add_many("FILE_WRITE", ["create file.txt", "write to notes.txt", "save as report.docx", "make a new file"])
add_many("FILE_DELETE", ["delete file.txt", "remove temp files", "erase the backup", "trash the document"])
add_many("FILE_LIST", ["list files", "show folder contents", "display directory", "what files are here"])

# ROLLBACK / UNKNOWN
add_many("ROLLBACK", [
    "undo that", "go back", "revert the change", "rollback the last action",
    "restore previous state", "cancel the last step", "undo the last thing"
])
add_many("UNKNOWN", [
    "blue banana quantum toaster", "maybe later maybe not", "the sky is square",
    "lorem ipsum help me", "noise words only", "xyzzy plugh", "random gibberish",
    "what even is this", "alpha beta gamma delta", "turn the thing into the stuff",
    "banana orbit tape", "wobble noodle seven"
])

# Hard overlap traps
OVERLAP_TRAPS = [
    ("chrome is slow", "CHAT"),
    ("i am looking at chrome", "CHAT"),
    ("do you like vscode", "CHAT"),
    ("the browser is open", "CHAT"),
    ("i am saving my file", "CHAT"),
    ("my goal is to learn python", "CHAT"),
    ("can you remember how to help me", "MEMORY_RECALL"),
    ("i want to forget this feeling", "CHAT"),
    ("what is on the screen right now", "READ_SCREEN"),
    ("show me the active window", "ACTIVE_WINDOW"),
    ("can you go to github and search for python", "WEB_NAVIGATE"),
    ("open chrome and search for news", "WEB_SEARCH"),
    ("please close the browser after you search", "CLOSE_APP"),
    ("remember the report and open it", "MEMORY_STORE"),
    ("list my windows and open chrome", "LIST_WINDOWS"),
    ("yes please go ahead", "CONFIRM"),
    ("no stop that", "CANCEL"),
    ("what should i do next", "REASONING"),
    ("who am i", "IDENTITY_QUERY"),
    ("are you awake", "CHAT"),
]
for text, label in OVERLAP_TRAPS:
    add(text, label)

WEAK_CLASS_BLOOM = {
    "CHAT": [
        "uh hi", "hey bro", "can we chat", "just talk to me",
        "tell me something useful", "this is weird", "i am tired",
        "answer normally", "make it simple", "be calm"
    ],
    "IDENTITY_QUERY": [
        "what exactly are you", "are you just a bot", "what is aeris made of",
        "how are you built", "who built you", "what is your architecture"
    ],
    "CONFIRM": [
        "yep go ahead", "yes continue", "okay do that", "sure proceed",
        "that works", "approved", "fine by me"
    ],
    "CANCEL": [
        "nope stop", "wait cancel that", "hold it", "pause everything",
        "never mind ignore it", "stop the action"
    ],
    "REASONING": [
        "help me choose", "what is the smarter choice", "what should we prioritize",
        "rank the options", "which path is best", "compare risk and reward"
    ],
    "MEMORY_RECALL": [
        "what do you know about me", "pull up the saved note",
        "do you still remember that", "what did you store",
        "retrieve what i said", "show me the stored context"
    ],
}
for label, texts in WEAK_CLASS_BLOOM.items():
    add_many(label, texts)

# Extra structured enrichment for the classes that were weakest in validation
CHAT_CONTEXTS = [
    "bro", "mate", "friend", "buddy", "boss", "man", "dude", "team",
    "hey", "hello", "aeris", "this", "that", "it", "something"
]
CHAT_FEELINGS = [
    "i am confused", "i am frustrated", "i am tired", "i am happy",
    "i am not sure", "i am in a hurry", "i need a quick answer",
    "make it simpler", "be direct", "be calm", "be detailed"
]
for prefix in ["uh", "hey", "yo", "okay", "well", "so", "actually", "seriously"]:
    for ctx in CHAT_CONTEXTS:
        add(f"{prefix} {ctx}", "CHAT")
for phrase in CHAT_FEELINGS:
    add(phrase, "CHAT")
    add(f"can you {phrase}", "CHAT")
    add(f"please {phrase}", "CHAT")

IDENTITY_FORMS = [
    "who are you", "what are you", "what exactly are you", "what is aeris",
    "what is your name", "what can you do", "what are your capabilities",
    "how are you built", "who built you", "what is your architecture",
    "are you human", "are you ai", "are you conscious", "are you a robot"
]
for base in IDENTITY_FORMS:
    add(base, "IDENTITY_QUERY")
    add(f"tell me {base}", "IDENTITY_QUERY")
    add(f"can you answer {base}", "IDENTITY_QUERY")
    add(f"i want to know {base}", "IDENTITY_QUERY")
    add(f"just curious, {base}", "IDENTITY_QUERY")

for base in ["yes", "yeah", "yep", "okay", "ok", "sure", "go ahead", "proceed"]:
    add(base, "CONFIRM")
    add(f"{base} please", "CONFIRM")
    add(f"yes {base}" if base in {"yes", "yeah", "yep"} else f"please {base}", "CONFIRM")
    add(f"that is {base}", "CONFIRM")

for base in ["no", "nope", "stop", "cancel", "abort", "wait", "pause", "hold on", "never mind"]:
    add(base, "CANCEL")
    add(f"{base} please", "CANCEL")
    add(f"please {base}", "CANCEL")
    add(f"i said {base}", "CANCEL")

REASONING_PROMPTS = [
    "should i choose speed or safety", "should i choose cost or quality",
    "what is the best approach", "what is the smarter choice",
    "compare the risks", "compare the options", "help me decide",
    "what should i prioritize", "what is the tradeoff", "rank the choices"
]
for prompt in REASONING_PROMPTS:
    add(prompt, "REASONING")
    add(f"can you {prompt}", "REASONING")
    add(f"please {prompt}", "REASONING")
    add(f"i need you to {prompt}", "REASONING")

MEMORY_RECALL_TOPICS = [
    "my preferences", "my project", "my last instruction", "my note",
    "my meeting", "my goal", "the context", "the saved information",
    "the stored note", "the current task", "the dashboard", "the browser"
]
for topic in MEMORY_RECALL_TOPICS:
    add(f"what do you remember about {topic}", "MEMORY_RECALL")
    add(f"do you remember {topic}", "MEMORY_RECALL")
    add(f"recall {topic}", "MEMORY_RECALL")
    add(f"pull up {topic}", "MEMORY_RECALL")
    add(f"what did you save about {topic}", "MEMORY_RECALL")

# Reduce command/chat confusion via explicit contrast examples
for app in ["chrome", "vscode", "browser", "terminal", "file explorer", "spotify", "discord"]:
    add(f"open {app}", "OPEN_APP")
    add(f"i am talking about {app}", "CHAT")
    add(f"the {app} is slow", "CHAT")
    add(f"can you check {app}", "CHAT")

INTENT_TRAINING_DATA = dedupe_pairs(INTENT_TRAINING_DATA)
INTENT_TRAINING_DATA = dedupe_pairs(INTENT_TRAINING_DATA)
random.shuffle(INTENT_TRAINING_DATA)

def get_stats() -> dict:
    counts = Counter(label for _, label in INTENT_TRAINING_DATA)
    return {
        "total": len(INTENT_TRAINING_DATA),
        "label_counts": dict(counts),
        "unique_inputs": len({t.lower() for t, _ in INTENT_TRAINING_DATA}),
    }
