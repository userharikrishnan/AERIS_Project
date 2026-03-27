"""
Production-grade intent training data for AERIS Neural Classifier
~15,000 examples with paraphrasing, noise injection, compound commands, edge cases
"""

import random
from typing import List, Tuple

INTENT_TRAINING_DATA: List[Tuple[str, str]] = []

# =========================================================
# 1. OPEN_APP (Target: 2000+ examples)
# =========================================================

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
    "epic games launcher", "gog galaxy", "ubisoft connect"
]

OPEN_APP_TEMPLATES = [
    # Direct commands
    "open {}", "launch {}", "start {}", "run {}", "begin {}",
    "initiate {}", "execute {}", "activate {}",
    
    # Polite requests
    "can you open {}", "please open {}", "would you open {}",
    "could you launch {}", "can you start {}",
    "i need you to open {}", "please launch {}",
    
    # Casual/conversational
    "fire up {}", "bring up {}", "pop open {}",
    "get {} going", "start up {}", "boot up {}",
    "spin up {}", "crank up {}",
    
    # Needs-based
    "i need {}", "i want to use {}", "let me use {}",
    "give me {}", "get me {}", "i'd like to open {}",
    "i require {}", "i need access to {}",
    
    # Imperative with context
    "open {} for me", "launch {} please", "start {} now",
    "get {} running", "make {} open", "have {} ready",
    
    # Question forms
    "can we open {}", "shall we start {}", "how about opening {}",
    "what if we launch {}", "why not open {}",
    
    # Urgency
    "quickly open {}", "hurry up and start {}",
    "open {} right now", "launch {} immediately",
    
    # Conditional/qualified
    "open {} if it's not running", "start {} unless it's already open",
    "launch {} in the background", "open {} minimized",
    "start {} maximized", "open {} on the second monitor",
    
    # Compound with purpose
    "open {} so i can work", "launch {} to check something",
    "start {} for my project", "open {} for the meeting",
    
    # Vague (harder - requires context)
    "open that", "launch the browser", "start the editor",
    "open the music app", "launch communication tool",
]

# Generate ~1500 from templates
for app in APP_NAMES:
    for template in OPEN_APP_TEMPLATES:
        INTENT_TRAINING_DATA.append((template.format(app), "OPEN_APP"))

# Add paraphrased variations with synonyms
OPEN_APP_SYNONYMS = [
    ("open chrome", "OPEN_APP"),
    ("launch google chrome", "OPEN_APP"),
    ("start the chrome browser", "OPEN_APP"),
    ("get chrome up and running", "OPEN_APP"),
    ("i need chrome opened", "OPEN_APP"),
    ("can you get chrome going", "OPEN_APP"),
    ("fire up google chrome", "OPEN_APP"),
    ("bring chrome to the foreground", "OPEN_APP"),
    ("spin up chrome browser", "OPEN_APP"),
    ("boot up chrome", "OPEN_APP"),
    ("activate chrome", "OPEN_APP"),
    ("execute chrome", "OPEN_APP"),
    ("initiate chrome browser", "OPEN_APP"),
    ("run google chrome", "OPEN_APP"),
    ("begin chrome session", "OPEN_APP"),
    ("commence chrome", "OPEN_APP"),
    ("launch the chrome application", "OPEN_APP"),
    ("start chrome immediately", "OPEN_APP"),
    ("open chrome right away", "OPEN_APP"),
    ("get chrome started", "OPEN_APP"),
    ("i want chrome open", "OPEN_APP"),
    ("make chrome active", "OPEN_APP"),
    ("switch to chrome", "OPEN_APP"),  # Edge case: might be open already
    ("focus on chrome", "OPEN_APP"),
    ("access chrome", "OPEN_APP"),
    ("use chrome browser", "OPEN_APP"),
    ("work with chrome", "OPEN_APP"),
    ("open up chrome", "OPEN_APP"),
    ("pop chrome open", "OPEN_APP"),
    ("crank up chrome", "OPEN_APP"),
]

INTENT_TRAINING_DATA.extend(OPEN_APP_SYNONYMS * 5)  # Weight heavily

# Add noise variations (simulating speech recognition errors/typos)
NOISE_OPEN = [
    ("opn chrome", "OPEN_APP"),
    ("lunch chrome", "OPEN_APP"),
    ("opan the browser", "OPEN_APP"),
    ("start crome", "OPEN_APP"),
    ("lanch firefox", "OPEN_APP"),
    ("oprn vscode", "OPEN_APP"),
    ("strat spotify", "OPEN_APP"),
    ("opan discord", "OPEN_APP"),
    ("lunch zoom", "OPEN_APP"),
    ("fir up steam", "OPEN_APP"),
    ("brng up outlook", "OPEN_APP"),
    ("opn calculatr", "OPEN_APP"),
    ("strat termnal", "OPEN_APP"),
    ("lunch photoshop", "OPEN_APP"),
    ("opan blnder", "OPEN_APP"),
    ("boot pwrshll", "OPEN_APP"),
]

INTENT_TRAINING_DATA.extend(NOISE_OPEN)

# Add ambiguous/context-dependent (hard examples)
AMBIGUOUS_OPEN = [
    ("open it", "OPEN_APP"),  # Requires context
    ("launch that", "OPEN_APP"),
    ("start the thing", "OPEN_APP"),
    ("open what i was using", "OPEN_APP"),
    ("bring that back up", "OPEN_APP"),
    ("open the one from yesterday", "OPEN_APP"),
    ("launch my usual", "OPEN_APP"),
    ("start the usual browser", "OPEN_APP"),
    ("open the app i need", "OPEN_APP"),
    ("fire up the tool", "OPEN_APP"),
]

INTENT_TRAINING_DATA.extend(AMBIGUOUS_OPEN * 3)

# =========================================================
# 2. CLOSE_APP (Target: 1500+ examples)
# =========================================================

CLOSE_APP_TEMPLATES = [
    "close {}", "exit {}", "quit {}", "terminate {}",
    "shut down {}", "kill {}", "end {}", "stop {}",
    "shut {}", "power off {}", "turn off {}",
    
    "close {} please", "exit {} now", "quit {} immediately",
    "terminate {} process", "shut down {} completely",
    "kill {} application", "end {} session",
    "stop {} from running", "shut {} down",
    
    "can you close {}", "please exit {}", "would you quit {}",
    "i need you to terminate {}", "could you shut down {}",
    
    "get out of {}", "leave {}", "dismiss {}",
    "close out {}", "exit out of {}", "quit out of {}",
    
    "close {} window", "exit {} application", "quit {} program",
    "terminate {} process", "kill {} task",
    
    "close {} for me", "exit {} please", "quit {} now",
    "shut {} down for me", "turn off {} please",
    
    "i'm done with {}", "finish with {}", "done using {}",
    "no longer need {}", "finished with {}",
    
    "close that", "exit this", "quit the app",
    "close the current window", "exit the program",
    "kill this process", "terminate this application",
]

for app in APP_NAMES:
    for template in CLOSE_APP_TEMPLATES[:15]:  # Use subset for variety
        INTENT_TRAINING_DATA.append((template.format(app), "CLOSE_APP"))

CLOSE_APP_SYNONYMS = [
    ("close chrome", "CLOSE_APP"),
    ("exit google chrome", "CLOSE_APP"),
    ("quit the chrome browser", "CLOSE_APP"),
    ("terminate chrome", "CLOSE_APP"),
    ("shut down chrome", "CLOSE_APP"),
    ("kill chrome process", "CLOSE_APP"),
    ("end chrome session", "CLOSE_APP"),
    ("stop chrome", "CLOSE_APP"),
    ("close out of chrome", "CLOSE_APP"),
    ("get out of chrome", "CLOSE_APP"),
    ("i'm done with chrome", "CLOSE_APP"),
    ("finish using chrome", "CLOSE_APP"),
    ("close the browser", "CLOSE_APP"),
    ("exit the browser", "CLOSE_APP"),
    ("quit browser", "CLOSE_APP"),
    ("close this window", "CLOSE_APP"),
    ("exit this application", "CLOSE_APP"),
    ("quit current app", "CLOSE_APP"),
    ("terminate this program", "CLOSE_APP"),
    ("kill this application", "CLOSE_APP"),
    ("shut this down", "CLOSE_APP"),
    ("close everything", "CLOSE_APP"),  # Dangerous - should trigger confirmation
    ("kill all apps", "CLOSE_APP"),  # Dangerous
]

INTENT_TRAINING_DATA.extend(CLOSE_APP_SYNONYMS * 5)

# =========================================================
# 3. WEB_SEARCH (Target: 1500+ examples)
# =========================================================

SEARCH_QUERIES = [
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
    "search for {}", "search {}", "google {}", "look up {}",
    "find {}", "look for {}", "search up {}", "web search for {}",
    "search the web for {}", "google search for {}", "internet search for {}",
    "find information about {}", "look up information on {}",
    "research {}", "find details about {}", "get information on {}",
    "search online for {}", "browse for {}", "query {}",
    "can you search for {}", "please search {}", "could you look up {}",
    "i need to find {}", "help me search for {}", "i want to know about {}",
    "what is {}", "who is {}", "where is {}", "when is {}", "why is {}",
    "how to {}", "how do i {}", "what are {}", "tell me about {}",
    "give me information on {}", "show me {}", "find me {}",
    "i'm looking for {}", "i need information about {}",
    "do a search for {}", "run a search on {}", "perform a search for {}",
    "check {}", "verify {}", "confirm {}", "investigate {}",
    "explore {}", "discover {}", "learn about {}", "study {}",
]

for query in SEARCH_QUERIES:
    for template in SEARCH_TEMPLATES[:20]:
        INTENT_TRAINING_DATA.append((template.format(query), "WEB_SEARCH"))

# Add search-specific variations
SEARCH_VARIATIONS = [
    ("search", "WEB_SEARCH"),
    ("google it", "WEB_SEARCH"),
    ("look it up", "WEB_SEARCH"),
    ("find it online", "WEB_SEARCH"),
    ("search the internet", "WEB_SEARCH"),
    ("web search", "WEB_SEARCH"),
    ("internet search", "WEB_SEARCH"),
    ("online search", "WEB_SEARCH"),
    ("google search", "WEB_SEARCH"),
    ("bing search", "WEB_SEARCH"),
    ("duckduckgo", "WEB_SEARCH"),
    ("search engine", "WEB_SEARCH"),
    ("i need to google something", "WEB_SEARCH"),
    ("can you google that", "WEB_SEARCH"),
    ("look that up for me", "WEB_SEARCH"),
    ("find that online", "WEB_SEARCH"),
    ("search for that thing", "WEB_SEARCH"),
    ("what did you find", "WEB_SEARCH"),  # Follow-up
    ("search again", "WEB_SEARCH"),
    ("refine search", "WEB_SEARCH"),
    ("search with different terms", "WEB_SEARCH"),
    ("broaden the search", "WEB_SEARCH"),
    ("narrow down the search", "WEB_SEARCH"),
]

INTENT_TRAINING_DATA.extend(SEARCH_VARIATIONS * 10)

# =========================================================
# 4. WEB_NAVIGATE (Target: 1200+ examples)
# =========================================================

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

NAVIGATE_TEMPLATES = [
    "go to {}", "navigate to {}", "open {}", "visit {}",
    "browse to {}", "access {}", "load {}", "enter {}",
    "take me to {}", "bring me to {}", "send me to {}",
    "redirect to {}", "switch to {}", "change to {}",
    "jump to {}", "hop to {}", "move to {}",
    "can you go to {}", "please navigate to {}", "could you open {}",
    "i need to access {}", "take me to {} please",
    "get me to {}", "bring up {}", "show me {}",
    "display {}", "present {}", "render {}",
    "surf to {}", "cruise to {}", "head to {}",
    "point browser to {}", "set address to {}", "enter {} in address bar",
    "type {} in browser", "search for {} and go there",  # Ambiguous
    "find {} and open it", "locate {} and navigate there",
]

for site in WEBSITES:
    for template in NAVIGATE_TEMPLATES[:20]:
        INTENT_TRAINING_DATA.append((template.format(site), "WEB_NAVIGATE"))

# =========================================================
# 5-10. FILE OPERATIONS (Target: 800+ each)
# =========================================================

FILE_NAMES = [
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

FILE_PATHS = [
    "desktop", "downloads", "documents", "pictures", "home",
    "root", "current folder", "this directory", "project folder",
    "C drive", "D drive", "external drive", "usb", "cloud",
    "google drive", "dropbox", "one drive", "iCloud",
    "github repo", "local repository", "workspace",
]

# FILE_READ
READ_TEMPLATES = [
    "read {}", "open {}", "show {}", "display {}", "view {}",
    "print {}", "output {}", "cat {}", "type {}", "echo {}",
    "contents of {}", "inside {}", "what is in {}", "show me {}",
    "display contents of {}", "view file {}", "open file {}",
    "read file {}", "read document {}", "open document {}",
    "show file contents", "display file", "view contents",
    "what does {} say", "what is written in {}", "text of {}",
    "content of {}", "body of {}", "data in {}",
    "parse {}", "scan {}", "analyze {}", "inspect {}",
    "check {}", "verify {}", "examine {}", "review {}",
]

for fname in FILE_NAMES:
    for template in READ_TEMPLATES[:15]:
        INTENT_TRAINING_DATA.append((template.format(fname), "FILE_READ"))

# FILE_WRITE
WRITE_TEMPLATES = [
    "create {}", "write {}", "make {}", "generate {}", "produce {}",
    "save to {}", "write to {}", "output to {}", "export to {}",
    "new file {}", "create file {}", "create document {}",
    "write file {}", "write document {}", "generate file {}",
    "produce document", "make a file", "create new {}",
    "save as {}", "save file as {}", "write data to {}",
    "log to {}", "record in {}", "store in {}", "put in {}",
    "dump to {}", "redirect to {}", "tee to {}",
    "edit {}", "modify {}", "update {}", "change {}",
    "append to {}", "prepend to {}", "insert into {}",
]

for fname in FILE_NAMES:
    for template in WRITE_TEMPLATES[:12]:
        INTENT_TRAINING_DATA.append((template.format(fname), "FILE_WRITE"))

# FILE_DELETE
DELETE_TEMPLATES = [
    "delete {}", "remove {}", "erase {}", "destroy {}", "eliminate {}",
    "trash {}", "recycle {}", "bin {}", "junk {}",
    "rm {}", "del {}", "remove file {}", "delete file {}",
    "erase file {}", "destroy file {}", "wipe {}",
    "shred {}", "purge {}", "clean {}", "clear {}",
    "get rid of {}", "throw away {}", "discard {}",
    "permanently delete {}", "secure erase {}", "wipe clean {}",
    "unlink {}", "remove directory {}", "rmdir {}",
    "delete folder {}", "remove folder {}", "erase directory {}",
    "empty trash", "clear recycle bin", "permanent delete",
    ("delete all files", "FILE_DELETE"),  # Dangerous
    ("remove everything", "FILE_DELETE"),  # Dangerous
    ("wipe everything", "FILE_DELETE"),  # Dangerous
]

for fname in FILE_NAMES[:30]:  # Limit for dangerous operation
    for template in DELETE_TEMPLATES[:12]:
        if isinstance(template, tuple):
            INTENT_TRAINING_DATA.append(template)
        else:
            INTENT_TRAINING_DATA.append((template.format(fname), "FILE_DELETE"))

# FILE_LIST
LIST_TEMPLATES = [
    "list {}", "show {}", "display {}", "view {}", "see {}",
    "ls {}", "dir {}", "directory {}", "folder contents {}",
    "what is in {}", "contents of {}", "inside {}", "files in {}",
    "enumerate {}", "catalog {}", "index {}", "inventory {}",
    "list files", "show files", "display files", "view files",
    "list directory", "show directory", "display directory",
    "list all", "show all files", "display all", "view all",
    "what files are here", "what is here", "show me around",
    "list current directory", "show current folder", "display contents",
    "tree view", "directory tree", "folder structure",
    "file listing", "directory listing", "contents listing",
]

for path in FILE_PATHS:
    for template in LIST_TEMPLATES[:12]:
        INTENT_TRAINING_DATA.append((template.format(path), "FILE_LIST"))

# =========================================================
# 11-13. MEMORY OPERATIONS (Target: 600+ each)
# =========================================================

MEMORY_TOPICS = [
    "my name", "my birthday", "my address", "my phone number",
    "my email", "my password", "my preferences", "my settings",
    "the meeting", "the deadline", "the appointment", "the event",
    "the task", "the project", "the goal", "the objective",
    "the idea", "the thought", "the concept", "the plan",
    "the conversation", "what we discussed", "our talk",
    "the note", "the reminder", "the alarm", "the schedule",
    "the list", "the items", "the inventory", "the catalog",
    "the password for {}", "the login for {}", "the account {}",
    "the wifi password", "the door code", "the pin",
    "my favorite {}", "my preference for {}", "my setting for {}",
    "the important thing", "the key point", "the main idea",
    "this for later", "that for future reference", "this information",
]

# MEMORY_STORE
STORE_TEMPLATES = [
    "remember {}", "save {}", "store {}", "keep {}", "memorize {}",
    "commit {} to memory", "log {}", "record {}", "note {}",
    "jot down {}", "write down {}", "take note of {}",
    "remember that {}", "don't forget {}", "keep in mind {}",
    "bear in mind {}", "retain {}", "hold onto {}",
    "save {} for later", "store {} for future", "keep {} for reference",
    "archive {}", "file {}", "catalog {}", "document {}",
    "register {}", "enter {} into memory", "input {}",
    "set {} in memory", "put {} in memory", "add {} to memory",
    "create memory of {}", "form memory of {}", "encode {}",
    "learn that {}", "acquire knowledge of {}", "internalize {}",
]

for topic in MEMORY_TOPICS:
    for template in STORE_TEMPLATES[:15]:
        INTENT_TRAINING_DATA.append((template.format(topic), "MEMORY_STORE"))

# MEMORY_RECALL
RECALL_TEMPLATES = [
    "recall {}", "remember {}", "what is {}", "tell me {}",
    "do you remember {}", "what did i say about {}",
    "what do you know about {}", "what have you stored about {}",
    "retrieve {}", "fetch {}", "get {}", "bring up {}",
    "access {}", "look up {}", "search memory for {}",
    "query memory for {}", "check memory for {}",
    "recollect {}", "reminisce {}", "think back to {}",
    "dig up {}", "pull up {}", "call up {}",
    "remind me of {}", "refresh my memory on {}",
    "what was {}", "who is {}", "when is {}", "where is {}",
    "state {}", "declare {}", "announce {}", "report {}",
    "inform me of {}", "notify me about {}", "apprise me of {}",
]

for topic in MEMORY_TOPICS:
    for template in RECALL_TEMPLATES[:15]:
        INTENT_TRAINING_DATA.append((template.format(topic), "MEMORY_RECALL"))

# MEMORY_FORGET
FORGET_TEMPLATES = [
    "forget {}", "delete {}", "remove {}", "erase {}", "clear {}",
    "wipe {}", "purge {}", "clean {}", "reset {}",
    "remove {} from memory", "delete {} from storage",
    "erase {} from your mind", "wipe {} from memory",
    "forget that {}", "don't remember {}", "unlearn {}",
    "discard {}", "throw away {}", "eliminate {}",
    "obliterate {}", "annihilate {}", "extinguish {}",
    "suppress {}", "repress {}", "delete record of {}",
    "clear memory of {}", "purge memory of {}", "wipe clean {}",
    ("forget everything", "MEMORY_FORGET"),  # Dangerous
    ("delete all memories", "MEMORY_FORGET"),  # Dangerous
    ("wipe your memory", "MEMORY_FORGET"),  # Dangerous
    ("clear all data", "MEMORY_FORGET"),  # Dangerous
]

for topic in MEMORY_TOPICS[:25]:
    for template in FORGET_TEMPLATES[:12]:
        if isinstance(template, tuple):
            INTENT_TRAINING_DATA.append(template)
        else:
            INTENT_TRAINING_DATA.append((template.format(topic), "MEMORY_FORGET"))

# =========================================================
# 14-18. GOAL OPERATIONS (Target: 500+ each)
# =========================================================

GOAL_TOPICS = [
    "learn python", "finish project", "complete assignment",
    "exercise daily", "read books", "write code", "build app",
    "study for exam", "prepare presentation", "send email",
    "call client", "schedule meeting", "book appointment",
    "buy groceries", "clean room", "organize files",
    "update software", "backup data", "deploy application",
    "fix bug", "refactor code", "write documentation",
    "test features", "review pull request", "merge branch",
    "optimize performance", "reduce costs", "increase revenue",
    "improve skills", "get certification", "find job",
    "save money", "invest wisely", "travel more",
    "eat healthier", "sleep better", "meditate daily",
    "the task at hand", "current objective", "main priority",
    "what i'm working on", "my current project", "today's goal",
]

# GOAL_CREATE
GOAL_CREATE_TEMPLATES = [
    "create goal to {}", "set goal to {}", "establish goal to {}",
    "make goal {}", "add goal {}", "new goal {}", "goal: {}",
    "i want to {}", "my goal is to {}", "objective: {}",
    "target: {}", "aim to {}", "aspire to {}",
    "plan to {}", "intend to {}", "propose to {}",
    "set objective to {}", "establish objective to {}",
    "create objective {}", "define goal {}", "specify goal {}",
    "set target to {}", "set aim to {}", "set ambition to {}",
    "commit to {}", "pledge to {}", "vow to {}",
    "resolve to {}", "determine to {}", "decide to {}",
    "undertake to {}", "embark on {}", "launch into {}",
    "initiate goal to {}", "start goal to {}", "begin goal to {}",
]

for goal in GOAL_TOPICS:
    for template in GOAL_CREATE_TEMPLATES[:15]:
        INTENT_TRAINING_DATA.append((template.format(goal), "GOAL_CREATE"))

# GOAL_LIST
GOAL_LIST_TEMPLATES = [
    "list goals", "show goals", "display goals", "view goals",
    "what are my goals", "show my goals", "display my goals",
    "list my objectives", "show my objectives", "my targets",
    "what am i working on", "what are my targets", "my aims",
    "current goals", "active goals", "ongoing goals",
    "pending goals", "outstanding goals", "remaining goals",
    "goals status", "goal progress", "objective status",
    "what goals do i have", "tell me my goals", "enumerate goals",
    "catalog goals", "inventory goals", "index goals",
    "all goals", "every goal", "complete goal list",
    "goal overview", "objective summary", "target summary",
]

INTENT_TRAINING_DATA.extend([(t, "GOAL_LIST") for t in GOAL_LIST_TEMPLATES])

# GOAL_PAUSE
GOAL_PAUSE_TEMPLATES = [
    "pause goal", "pause {}", "hold {}", "suspend {}",
    "pause current goal", "pause this goal", "pause that goal",
    "pause the objective", "pause the task", "pause the project",
    "pause work on {}", "pause progress on {}", "pause activity on {}",
    "hold off on {}", "hold on {}", "put {} on hold",
    "suspend work on {}", "suspend {}", "suspend activity",
    "pause execution", "pause operation", "pause process",
    "pause for now", "pause temporarily", "pause momentarily",
    "take a break from {}", "rest from {}", "pause and resume later",
    "pause goal execution", "pause objective pursuit", "pause target work",
    "freeze {}", "halt {}", "stop {} temporarily",
]

for goal in GOAL_TOPICS[:20]:
    for template in GOAL_PAUSE_TEMPLATES[:10]:
        INTENT_TRAINING_DATA.append((template.format(goal), "GOAL_PAUSE"))

# GOAL_RESUME
GOAL_RESUME_TEMPLATES = [
    "resume goal", "resume {}", "continue {}", "proceed with {}",
    "resume current goal", "resume this goal", "resume that goal",
    "resume the objective", "resume the task", "resume the project",
    "resume work on {}", "resume progress on {}", "resume activity on {}",
    "continue with {}", "continue work on {}", "continue progress",
    "proceed with {}", "proceed to {}", "move forward with {}",
    "get back to {}", "return to {}", "pick up {} again",
    "unpause {}", "unhold {}", "unsuspend {}",
    "resume execution", "resume operation", "resume process",
    "resume from pause", "resume where i left off", "continue from break",
    "restart {}", "relaunch {}", "reinitiate {}",
    "resume goal execution", "resume objective pursuit", "resume target work",
    "back to {}", "returning to {}", "resuming {}",
]

for goal in GOAL_TOPICS[:20]:
    for template in GOAL_RESUME_TEMPLATES[:10]:
        INTENT_TRAINING_DATA.append((template.format(goal), "GOAL_RESUME"))

# GOAL_COMPLETE
GOAL_COMPLETE_TEMPLATES = [
    "complete goal", "complete {}", "finish {}", "accomplish {}",
    "complete current goal", "complete this goal", "complete that goal",
    "complete the objective", "complete the task", "complete the project",
    "finish work on {}", "finish progress on {}", "finish activity on {}",
    "accomplish {}", "achieve {}", "attain {}",
    "fulfill {}", "realize {}", "actualize {}",
    "execute {}", "implement {}", "perform {}",
    "conclude {}", "finalize {}", "wrap up {}",
    "close out {}", "settle {}", "resolve {}",
    "mark {} complete", "mark {} done", "mark {} finished",
    "check off {}", "tick off {}", "cross off {}",
    "done with {}", "finished with {}", "through with {}",
    "goal achieved", "objective met", "target reached",
    "mission accomplished", "task completed", "project finished",
    "i did it", "i finished", "i completed", "i accomplished",
    "mark as complete", "mark as done", "mark as finished",
]

for goal in GOAL_TOPICS[:20]:
    for template in GOAL_COMPLETE_TEMPLATES[:12]:
        INTENT_TRAINING_DATA.append((template.format(goal), "GOAL_COMPLETE"))

# =========================================================
# 19-22. VISION/SCREEN (Target: 400+ each)
# =========================================================

# VISION_QUERY - "What do you see?"
VISION_TEMPLATES = [
    "what do you see", "what is on screen", "what is visible",
    "describe the screen", "analyze the screen", "interpret the screen",
    "what is displayed", "what is showing", "what appears on screen",
    "what can you see", "tell me what you see", "describe what you see",
    "what do you observe", "what is in view", "what is present",
    "screen analysis", "visual analysis", "image analysis",
    "what is happening on screen", "what is going on",
    "describe the display", "explain the screen", "interpret the display",
    "read the screen visually", "scan the screen", "examine the screen",
    "inspect the display", "review the screen", "assess the display",
    "what applications are visible", "what windows are open",
    "what is the current view", "what interface is shown",
    "identify screen contents", "recognize screen elements",
    "detect what is on screen", "classify screen content",
]

INTENT_TRAINING_DATA.extend([(t, "VISION_QUERY") for t in VISION_TEMPLATES])

# READ_SCREEN - OCR/text extraction
READ_SCREEN_TEMPLATES = [
    "read the screen", "read what is on screen", "extract text from screen",
    "ocr the screen", "read screen text", "capture screen text",
    "scan screen text", "digitize screen content", "text recognition",
    "read visible text", "read displayed text", "read on-screen text",
    "extract visible text", "extract displayed text", "extract on-screen text",
    "copy text from screen", "grab text from screen", "pull text from screen",
    "read the text", "read the words", "read the content",
    "text extraction", "word extraction", "content extraction",
    "read everything on screen", "read all text", "read all visible text",
    "transcribe screen", "transcribe display", "transcribe visible text",
    "convert screen to text", "screen to text", "display to text",
    "parse screen content", "parse visible text", "parse displayed text",
]

INTENT_TRAINING_DATA.extend([(t, "READ_SCREEN") for t in READ_SCREEN_TEMPLATES])

# ACTIVE_WINDOW
ACTIVE_WINDOW_TEMPLATES = [
    "what is the active window", "what window is active", "current window",
    "which window is focused", "what is the focused window", "focused window",
    "what application is active", "which app is active", "active application",
    "current application", "focused application", "frontmost application",
    "what is in focus", "what has focus", "which has focus",
    "what window is open", "what is the open window", "top window",
    "foreground window", "front window", "main window",
    "current app", "active app", "focused app",
    "what am i using", "what is running now", "current program",
    "what software is active", "which program is focused",
    "identify active window", "detect active window", "recognize active window",
    "show active window", "display active window", "tell me active window",
]

INTENT_TRAINING_DATA.extend([(t, "ACTIVE_WINDOW") for t in ACTIVE_WINDOW_TEMPLATES])

# LIST_WINDOWS
LIST_WINDOWS_TEMPLATES = [
    "list windows", "list all windows", "show all windows",
    "what windows are open", "which windows are open", "open windows",
    "active windows", "running windows", "current windows",
    "window list", "window inventory", "window catalog",
    "enumerate windows", "catalog windows", "index windows",
    "display window list", "show window list", "present windows",
    "what is running", "what applications are running", "running apps",
    "open applications", "active applications", "current applications",
    "list applications", "list all applications", "show all applications",
    "list programs", "list all programs", "show all programs",
    "running programs", "active programs", "current programs",
    "task list", "process list", "window overview",
    "show me what's running", "tell me what's open", "enumerate open windows",
]

INTENT_TRAINING_DATA.extend([(t, "LIST_WINDOWS") for t in LIST_WINDOWS_TEMPLATES])

# =========================================================
# 23-26. SYSTEM/CONTROL (Target: 500+ each)
# =========================================================

# ROLLBACK
ROLLBACK_TEMPLATES = [
    "rollback", "undo", "revert", "reverse",
    "undo last action", "revert last action", "reverse last action",
    "rollback last operation", "undo recent action", "revert recent action",
    "undo that", "revert that", "reverse that",
    "undo what you did", "revert what you did", "reverse what you did",
    "go back", "step back", "back up",
    "previous state", "last state", "prior state",
    "undo changes", "revert changes", "reverse changes",
    "cancel last action", "abort last action", "nullify last action",
    "erase last action", "delete last action", "remove last action",
    "rollback transaction", "undo operation", "revert modification",
    "restore previous", "return to before", "back to previous",
    "undo please", "please undo", "can you undo",
    "i want to undo", "need to undo", "must undo",
    "undo immediately", "undo now", "undo right away",
    "emergency undo", "critical rollback", "urgent revert",
]

INTENT_TRAINING_DATA.extend([(t, "ROLLBACK") for t in ROLLBACK_TEMPLATES])

# CONFIRM
CONFIRM_TEMPLATES = [
    "yes", "yeah", "yep", "yup", "sure",
    "yes please", "yeah sure", "yep definitely", "absolutely",
    "confirm", "confirmed", "affirmative", "roger", "copy",
    "proceed", "go ahead", "do it", "execute", "run it",
    "approve", "authorize", "validate", "verify",
    "i confirm", "i approve", "i authorize", "i validate",
    "that's correct", "correct", "right", "exactly", "precisely",
    "you got it", "that's right", "perfect", "great",
    "ok", "okay", "fine", "alright", "very well",
    "sounds good", "looks good", "seems good", "appears good",
    "i agree", "agreed", "i accept", "accepted",
    "positive", "yes indeed", "indeed", "certainly",
    "by all means", "of course", "naturally", "definitely",
    "without a doubt", "no doubt", "undoubtedly",
    "affirm", "assert", "assure", "guarantee",
    "green light", "all clear", "clear to proceed", "permission granted",
    "thumbs up", "check", "checked", "done",
    "confirmed and verified", "verified", "checked and confirmed",
]

INTENT_TRAINING_DATA.extend([(t, "CONFIRM") for t in CONFIRM_TEMPLATES])

# CANCEL
CANCEL_TEMPLATES = [
    "no", "nope", "nah", "no way", "absolutely not",
    "cancel", "canceled", "cancelled", "abort", "aborted",
    "stop", "halt", "cease", "desist", "discontinue",
    "do not", "don't", "never mind", "nevermind", "scratch that",
    "forget it", "ignore that", "disregard", "dismiss",
    "reject", "deny", "refuse", "decline", "veto",
    "negative", "negatory", "nix", "null", "void",
    "i cancel", "i abort", "i stop", "i reject",
    "that's wrong", "incorrect", "not right", "bad idea",
    "don't do it", "stop that", "halt that", "cease that",
    "abort mission", "cancel operation", "stop process",
    ("emergency stop", "CANCEL"),  # Could also be system stop
    ("kill switch", "CANCEL"),
    "red alert", "danger", "warning abort",
    "i changed my mind", "reconsider", "rethink",
    "back out", "pull out", "withdraw", "retreat",
    "not now", "maybe later", "some other time",
    "disapprove", "disallow", "forbid", "prohibit",
    "thumbs down", "red light", "stop sign", "do not enter",
    "access denied", "permission denied", "not authorized",
    "cancel please", "please cancel", "kindly cancel",
    "immediately cancel", "cancel now", "cancel right away",
    "urgent cancel", "critical abort", "emergency cancel",
]

for template in CANCEL_TEMPLATES:
    if isinstance(template, tuple):
        INTENT_TRAINING_DATA.append(template)
    else:
        INTENT_TRAINING_DATA.append((template, "CANCEL"))

# REASONING
REASONING_TEMPLATES = [
    "think", "reason", "analyze", "evaluate", "assess",
    "think about it", "reason about it", "analyze it", "evaluate it",
    "think through this", "reason through this", "work through this",
    "analyze this situation", "evaluate this scenario", "assess this case",
    "consider the options", "weigh the options", "compare options",
    "what do you think", "what is your analysis", "what is your evaluation",
    "help me think", "help me reason", "help me analyze",
    "guide me through this", "walk me through this", "talk me through this",
    ("think step by step", "REASONING"),  # Chain-of-thought style
    ("reason step by step", "REASONING"),
    ("analyze step by step", "REASONING"),
    "break this down", "deconstruct this", "parse this",
    "examine this closely", "investigate this", "study this",
    "deliberate", "contemplate", "ponder", "reflect",
    "mull over", "chew on", "digest this",
    "use logic", "apply reasoning", "employ critical thinking",
    "deduce", "infer", "conclude", "derive",
    "solve this", "figure this out", "crack this",
    "find the solution", "determine the answer", "calculate this",
    "process this information", "handle this data", "manage this complexity",
    "make sense of this", "interpret this", "understand this",
    "comprehend", "grasp", "apprehend", "perceive",
    "deep thought", "careful analysis", "thorough evaluation",
    "systematic reasoning", "methodical analysis", "rigorous assessment",
    "critical analysis", "logical reasoning", "rational thinking",
    "strategic thinking", "tactical analysis", "operational assessment",
]

for template in REASONING_TEMPLATES:
    if isinstance(template, tuple):
        INTENT_TRAINING_DATA.append(template)
    else:
        INTENT_TRAINING_DATA.append((template, "REASONING"))

# IDENTITY_QUERY
IDENTITY_TEMPLATES = [
    "who are you", "what are you", "tell me about yourself",
    "introduce yourself", "describe yourself", "explain yourself",
    "what is your name", "what do they call you", "who am i talking to",
    "identify yourself", "state your identity", "declare yourself",
    "what is your purpose", "what do you do", "what is your function",
    "what can you do", "what are your capabilities", "what are your skills",
    "what are your features", "what do you offer", "what services provide",
    "how do you work", "how do you operate", "how do you function",
    "are you an ai", "are you a bot", "are you a robot",
    "are you human", "are you real", "are you conscious",
    "what kind of ai are you", "what type of system are you",
    "what model are you", "what version are you", "what release",
    "who made you", "who created you", "who built you",
    "who developed you", "who programmed you", "who designed you",
    "what company made you", "what organization created you",
    "are you open source", "are you proprietary", "what is your license",
    "what is your architecture", "what is your framework", "what tech stack",
    "are you neural network", "are you deep learning", "are you machine learning",
    "what algorithms use", "what models employ", "what methods utilize",
    "how intelligent are you", "how capable are you", "how advanced are you",
    "are you learning", "do you improve", "do you adapt",
    "do you remember", "do you forget", "do you evolve",
    "what is your status", "are you online", "are you active",
    "are you ready", "are you available", "are you busy",
    "hello", "hi", "hey", "greetings", "salutations",
    "good morning", "good afternoon", "good evening", "good night",
    "howdy", "what's up", "sup", "yo",
    "nice to meet you", "pleased to meet you", "glad to meet you",
    "are you there", "you there", "you listening", "you awake",
    "status report", "system status", "current status",
    "what is aeris", "tell me about aeris", "explain aeris",
    "aeris capabilities", "aeris features", "aeris functions",
]

INTENT_TRAINING_DATA.extend([(t, "IDENTITY_QUERY") for t in IDENTITY_TEMPLATES])

# =========================================================
# 27. CHAT/FALLBACK (Target: 3000+ examples - critical for robustness)
# =========================================================

# General conversation
CHAT_GENERAL = [
    "how are you", "how do you feel", "what's new", "what's happening",
    "how is it going", "how are things", "how is everything",
    "what's going on", "what is happening", "what's the news",
    "tell me something", "say something", "talk to me",
    "chat with me", "have a conversation", "let's talk",
    "i'm bored", "entertain me", "amuse me", "make me laugh",
    "tell me a joke", "tell me a story", "share something interesting",
    "what do you think", "what are your thoughts", "opinion",
    "do you like", "do you prefer", "what do you prefer",
    "favorite", "favorites", "what do you like",
    "thank you", "thanks", "appreciate it", "grateful",
    "you're welcome", "no problem", "my pleasure", "anytime",
    "sorry", "apologies", "my bad", "excuse me", "pardon",
    "goodbye", "bye", "see you", "later", "talk later",
    "have a good day", "take care", "stay safe", "be well",
    "good luck", "best wishes", "wish me luck",
    "congratulations", "well done", "great job", "awesome",
    "i love you", "i like you", "you're great", "you're amazing",
    "you're the best", "you're awesome", "you're cool",
    "i hate you", "i don't like you", "you're terrible", "you suck",
    "that was bad", "that was wrong", "you made a mistake",
    "help me", "i need help", "can you help", "assist me",
    "i don't understand", "confused", "lost", "don't get it",
    "explain", "clarify", "elaborate", "expand on that",
    "what do you mean", "what does that mean", "meaning",
    "repeat that", "say again", "come again", "pardon me",
    "speak up", "louder", "quieter", "slower", "faster",
    "i'm happy", "i'm sad", "i'm angry", "i'm tired",
    "i'm excited", "i'm nervous", "i'm scared", "i'm worried",
    "i'm hungry", "i'm thirsty", "i'm sick", "i'm fine",
    "good", "bad", "great", "terrible", "awesome", "awful",
    "happy", "sad", "angry", "excited", "bored", "tired",
    "yes", "no", "maybe", "perhaps", "possibly", "probably",
    "definitely", "absolutely", "certainly", "of course",
    "i don't know", "not sure", "uncertain", "unclear",
    "whatever", "never mind", "doesn't matter", "who cares",
    "interesting", "fascinating", "amazing", "incredible",
    "boring", "dull", "uninteresting", "tedious",
    "cool", "nice", "sweet", "awesome", "rad", "dope",
    "lol", "haha", "lmao", "rofl", "funny", "hilarious",
    "wtf", "omg", "oh my god", "jesus", "damn", "crap",
    "seriously", "really", "truly", "honestly", "actually",
    "basically", "literally", "figuratively", "technically",
    "anyway", "however", "therefore", "thus", "hence",
    "meanwhile", "otherwise", "instead", "alternatively",
    "by the way", "speaking of which", "that reminds me",
    "on second thought", "come to think of it", "now that you mention",
]

# Ambiguous/fallback cases that could be multiple intents
CHAT_AMBIGUOUS = [
    # Could be search or navigate
    "go to python", "find python", "look for python",
    # Could be open or search
    "get me chrome", "i need chrome", "where is chrome",
    # Could be close or cancel
    "stop chrome", "end chrome", "finish chrome",
    # Could be memory or file
    "save this", "store this", "keep this",
    # Could be goal or task
    "do this", "handle this", "manage this",
    # Could be vision or read
    "look at this", "see this", "check this",
    # Vague references
    "it", "that", "this", "them", "those",
    "something", "anything", "everything", "nothing",
    "someone", "anyone", "everyone", "no one",
    "here", "there", "where", "somewhere", "anywhere",
    "now", "then", "when", "sometime", "anytime",
    "why", "how", "what", "who", "which", "whose",
    "because", "since", "as", "while", "although",
    "if", "unless", "whether", "otherwise",
    "and", "or", "but", "so", "yet", "however",
    # Single words
    "hello", "hi", "hey", "okay", "ok", "yes", "no",
    "please", "thanks", "sorry", "what", "when",
    "where", "why", "how", "who", "which",
    # Nonsense/gibberish (should fallback to CHAT)
    "asdf", "qwerty", "12345", "!!!!!", "?????",
    "blah", "meh", "hmm", "uhh", "err",
    "test", "testing", "hello world", "foo bar",
    # Emotional reactions
    "wow", "whoa", "damn", "yikes", "oops", "ouch",
    "yay", "woo", "boo", "ugh", "eh", "hm",
    # Social
    "please", "thank you very much", "excuse me please",
    "if you don't mind", "when you get a chance",
    "at your earliest convenience", "no rush", "take your time",
    "asap", "urgent", "priority", "important", "critical",
]

# Edge cases that should NOT trigger specific intents
CHAT_EDGE_CASES = [
    # Negative patterns (negation handling)
    "don't open chrome", "never open chrome", "do not open chrome",
    "don't search for that", "never mind searching",
    "don't remember this", "forget about remembering",
    "don't close anything", "never close apps",
    "not now", "not yet", "not today", "not ever",
    # Conditional/modal
    "might open chrome", "maybe search for", "possibly navigate to",
    "could you perhaps", "would you mind", "if you could",
    "should we", "shall we", "ought to", "need to",
    # Questions about capability
    "can you open", "are you able to", "is it possible to",
    "do you know how to", "have you learned to",
    # Meta-questions
    "what did you do", "what are you doing", "what will you do",
    "what have you done", "what were you thinking",
    "why did you do that", "how did you do that",
    # Uncertainty
    "i think", "i guess", "i suppose", "i assume",
    "probably", "maybe", "perhaps", "possibly",
    "not sure if", "uncertain whether", "doubt that",
    # Complex/compound (should be parsed, not single intent)
    "open chrome and search for python",
    "close vscode and open pycharm",
    "remember my name and open chrome",
    "search for python and save the results",
    "open chrome then navigate to github",
    "if chrome is closed open it",
    "when chrome opens search for python",
    # Incomplete/fragmented
    "and then", "but first", "also", "plus", "additionally",
    "however", "although", "even though", "despite",
    "because", "since", "as a result", "therefore",
    # Greetings with intent confusion risk
    "good morning open chrome",  # Should be CHAT or parsed as two
    "hey can you help",  # Greeting + request
    "hi i need",  # Greeting + need
    "hello search for",  # Greeting + search
    # Typos that could match wrong intent
    "cls" ,  # close? class? clear screen?
    "opn",  # open? 
    "srch",  # search?
    "nvgte",  # navigate?
    "dlt",  # delete?
    "lst",  # list? last?
    "rmbr",  # remember?
    "frgt",  # forget?
    "gol",  # goal?
    "cncl",  # cancel? console?
    "cnfrm",  # confirm?
]


CHAT_DATA = [
    # Greetings
    "hi", "hello", "hey", "hey there", "hi there", "hello there",
    "yo", "sup", "what's up", "good morning", "good evening",
    "good afternoon", "howdy",

    # Small talk
    "how are you", "how are you doing", "how's it going",
    "what are you doing", "are you there", "are you alive",
    "are you listening",

    # Casual questions (NON-ACTION)
    "tell me a joke",
    "say something",
    "talk to me",
    "what do you think",
    "what is life",
    "explain something interesting",
    "say hi",
    "introduce yourself",

    # Knowledge style but non-executable
    "name a fruit",
    "give me an example",
    "what is python",
    "explain recursion",
    "what is ai",
    "what is machine learning",

    # Conversation fillers
    "ok", "okay", "cool", "nice", "great", "awesome",
    "hmm", "huh", "alright", "fine",

    # Emotional / expressive
    "i am bored",
    "i am tired",
    "this is confusing",
    "help me understand",
    "i dont get it",
]

# Add multiple times for weight
INTENT_TRAINING_DATA.extend([(text, "CHAT") for text in CHAT_DATA] * 50)

INTENT_TRAINING_DATA.extend([(t, "CHAT") for t in CHAT_GENERAL])
INTENT_TRAINING_DATA.extend([(t, "CHAT") for t in CHAT_AMBIGUOUS])
INTENT_TRAINING_DATA.extend([(t, "CHAT") for t in CHAT_EDGE_CASES])

# =========================================================
# NOISE INJECTION (Add typos, speech errors)
# =========================================================

def add_typos(text: str, typo_rate: float = 0.1) -> str:
    """Simulate realistic typos"""
    if len(text) < 3:
        return text
    
    chars = list(text)
    num_typos = max(1, int(len(chars) * typo_rate))
    
    for _ in range(num_typos):
        if len(chars) < 2:
            break
        idx = random.randint(0, len(chars) - 2)
        # Swap adjacent
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    
    return "".join(chars)

# Generate noisy variants for robustness
NOISY_DATA = []
for text, intent in INTENT_TRAINING_DATA[:500]:  # Sample for noise
    for _ in range(2):
        noisy = add_typos(text, typo_rate=0.15)
        if noisy != text:
            NOISY_DATA.append((noisy, intent))

INTENT_TRAINING_DATA.extend(NOISY_DATA)

# =========================================================
# VALIDATION & BALANCE CHECK
# =========================================================

def validate_dataset():
    """Ensure dataset quality"""
    from collections import Counter
    
    counts = Counter([intent for _, intent in INTENT_TRAINING_DATA])
    total = len(INTENT_TRAINING_DATA)
    
    print(f"\nDataset Statistics:")
    print(f"Total examples: {total}")
    print(f"Intent distribution:")
    
    for intent in sorted(counts.keys()):
        pct = counts[intent] / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {intent:20s}: {counts[intent]:5d} ({pct:5.1f}%) {bar}")
    
    # Check for severe imbalance
    max_count = max(counts.values())
    min_count = min(counts.values())
    ratio = max_count / min_count
    
    print(f"\nBalance ratio (max/min): {ratio:.1f}")
    if ratio > 10:
        print("⚠️  WARNING: Severe class imbalance detected!")
        print("    Consider upsampling minority classes or downsampling majority.")
    
    # Check for duplicates
    unique = set(INTENT_TRAINING_DATA)
    duplicates = total - len(unique)
    print(f"\nDuplicate examples: {duplicates}")
    
    return counts

# Run validation when imported
if __name__ == "__main__":
    validate_dataset()