# Using `ratio_study` with Microsoft Copilot 365

A short guide. **Copilot 365 will walk you through every step.**

> **TL;DR:** Ask your IT person to create a Python **virtual environment** and
> install `ratio_study` inside it (never in the system Python). Then open
> Copilot 365, paste the **Setup Prompt** below, and let Copilot do the rest.

---

## 1. Ask IT to set up a virtual environment

Installing Python packages into the **system Python** can break your
operating system or other apps. We'll create an **isolated virtual
environment** (often called a "venv") and install `ratio_study` inside it.
Nothing leaves the venv, and you can delete it cleanly later.

Email or message your IT support person and ask them to run these
commands in a Terminal (Mac/Linux) or Command Prompt (Windows):

### macOS / Linux

```bash
# 1. Create a folder to hold the venv (anywhere you like)
mkdir -p ~/ratio_study_env
cd ~/ratio_study_env

# 2. Create the virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate

# 4. Upgrade pip and install ratio_study inside the venv
python -m pip install --upgrade pip
python -m pip install "ratio_study[mcp]"

# 5. Confirm the version
python -c "from ratio_study._version import __version__; print('version:', __version__)"
```

### Windows

```bat
:: 1. Create a folder to hold the venv (anywhere you like)
mkdir %USERPROFILE%\ratio_study_env
cd %USERPROFILE%\ratio_study_env

:: 2. Create the virtual environment
python -m venv .venv

:: 3. Activate it
.venv\Scripts\activate

:: 4. Upgrade pip and install ratio_study inside the venv
python -m pip install --upgrade pip
python -m pip install "ratio_study[mcp]"

:: 5. Confirm the version
python -c "from ratio_study._version import __version__; print('version:', __version__)"
```

> 💡 **Tell your IT person:** "If Python isn't installed, please install
> **Python 3.11 or 3.12** first from [python.org/downloads](https://www.python.org/downloads/).
> On Windows, tick **'Add Python to PATH'** during install."

When they're done, ask them to send you the **full path to the venv's
Python executable** — you'll need it for the Copilot connector. It'll
look something like:

- **Mac/Linux:** `/Users/yourname/ratio_study_env/.venv/bin/python`
- **Windows:** `C:\Users\yourname\ratio_study_env\.venv\Scripts\python.exe`

---

## 2. Open Copilot 365 and paste the Setup Prompt

Go to [copilot.microsoft.com](https://copilot.microsoft.com) and start a new
chat. Copy and paste the **entire** block below, then press **Enter**:

````
You are my ratio_study setup assistant. I have a `ratio_study` package
installed on my computer inside a Python virtual environment (a venv),
and it exposes tools via MCP (Model Context Protocol). I need you to
help me connect you to it.

I will give you the path to my venv's Python executable when needed.
It is the python *inside* the .venv folder, NOT the system python.
Do not run any pip install or python commands against the system
python — only operate inside my venv.

Please do the following, one step at a time:

1. Ask me for the full path to my venv's Python executable.
   - On Mac/Linux it usually ends in `.venv/bin/python`.
   - On Windows it usually ends in `.venv\Scripts\python.exe`.
   Save this path; you'll use it as the connector's "Command" value.

2. Ask me to open a Terminal / Command Prompt, activate the venv by
   running the activation command for my OS, and then run:
     <PYTHON_PATH> -m ratio_study.mcp_server
   to confirm the server starts. They should see no errors and the
   cursor should just sit there — that's a good sign. They press
   Ctrl+C to stop it.

3. Ask me to add a new MCP connector in Copilot 365. The fields are
   usually under Settings → Connectors → Add custom connector. The
   exact menu names vary, so help me find the right place if I
   can't see it. The values to enter are:
     - Name: ratio_study (or anything I like)
     - Description: Run IAAO-compliant ratio studies on local data files
     - Type: MCP server (or stdio if asked)
     - Command: <the full venv python path I gave you in step 1>
     - Arguments: -m ratio_study.mcp_server
   Then click Save / Test connection and confirm it says "Connected".

4. Ask me to start a NEW chat (so the connector loads) and paste this:
   "Use the ratio_study tool to list datasets in <the folder where
   I keep my data>." Help me pick that folder.

5. Remind me that my data file must be in one of these formats:
   CSV, TSV, Parquet, Excel (.xlsx/.xls), JSON, or SPSS (.sav).
   The full path to my file must be absolute (e.g.
   `C:\Users\me\data\sales.csv` on Windows or
   `/Users/me/data/sales.csv` on Mac).

6. Tell me about the four data tools I now have access to:
   - list_datasets — find files in a folder
   - inspect_dataset — peek at columns before analysing
   - run_ratio_study — the main one
   - get_module_capabilities — what indicators are available

7. Offer to do a test run on a file I have. If I say yes, ask me
   for the file path, then call inspect_dataset on it, show me
   what columns it found, and ask me which one is the "market
   value" (sale price) and which is the "assessment" (assessed
   value). Then ask which indicators I want and call run_ratio_study.

IMPORTANT RULES:
- If the venv python path doesn't work when added to the connector,
  help me debug. The most common cause is using the SYSTEM python
  path instead of the venv one. Run a quick check: ask me to open
  a terminal, activate the venv, and run
  `<PYTHON_PATH> -c "import ratio_study; print(ratio_study.__version__)"`
  If that prints a version, the venv python is correct.
- If the user wants to install or update anything, ONLY operate
  inside the activated venv. Never use system pip / system python.
- Keep your tone friendly, brief, and step-by-step. Do not try to
  run ratio_study on any data until you've confirmed with me
  which columns are which.
````

Copilot will then guide you through the rest.

---

## 3. (Optional) Quick prompts to use later

Once everything is set up, here are short prompts you can paste in any
new Copilot chat. **Replace the parts in `< >` with your own values.**

### Inspect a file
````
Use ratio_study to inspect `<full path to my file>` and tell me the
columns and their data types.
````

### Run a basic ratio study
````
Use ratio_study to run a basic study on `<file path>`. The market
value column is `<sale_price>` and the assessment column is
`<assessed_value>`. Include Median, Mean, CoD, PRD, PRB, and VEI.
````

### Run by neighborhood / group
````
Run a ratio study on `<file path>` with market value `<col>`, assessment
`<col>`, grouped by `<neighborhood>`. Include everything.
````

### Open an SPSS file
````
Open my SPSS file `<full path to .sav>`, show me the columns, and
then run a full ratio study (market value: `<col>`, assessment: `<col>`).
````

### Compliance check
````
Run a ratio study on `<file path>` (market: `<col>`, assessment:
`<col>`) grouped by `<col>` and tell me which groups are out of
compliance with IAAO standards.
````

### Update to a newer version
````
I have a newer version of ratio_study. Help me update the package
inside my virtual environment without breaking anything.
````
*(Copilot will walk you through activating the venv, then running
`python -m pip install --upgrade "ratio_study[mcp]"`.)*

### Quick diagnosis if something breaks
````
Call the `diagnose_setup` tool and tell me what's wrong. Remember
that the package is installed in a virtual environment, not the
system python.
````

---

## Troubleshooting (one-liners)

If something goes wrong, paste any of these into Copilot and it will
help you fix it. **All of these assume you're operating inside your
virtual environment.**

- *"Run diagnose_setup and tell me what's broken."*
- *"I'm getting 'Python is not recognized' — help me fix it. (Remember
  I'm using a venv, not the system python.)"*
- *"Copilot says the tool isn't available — what do I check?"*
- *"My file path isn't being found — help me get the right path."*
- *"The file is too large — help me shrink it."*
- *"How do I activate my virtual environment on [Mac/Windows]?"*
- *"How do I update ratio_study to the latest version inside my venv?"*
- *"How do I cleanly remove the venv and start over?"*

For anything else, just describe the error message in plain English
and Copilot will guide you.

---

## Appendix: How to remove everything (cleanup)

If you ever want to wipe `ratio_study` and the venv completely off your
machine:

### macOS / Linux

```bash
# Deactivate the venv if it's active
deactivate

# Delete the venv folder
rm -rf ~/ratio_study_env
```

### Windows

```bat
:: Deactivate the venv if it's active
deactivate

:: Delete the venv folder
rmdir /s /q %USERPROFILE%\ratio_study_env
```

Then go to Copilot 365 → Settings → Connectors and remove the
`ratio_study` connector. Done — nothing left behind.

---

*Last updated: 2026-06-12*
