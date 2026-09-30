# Copilot prompt template — run a ratio study without writing Python

For assessment professionals who know ratio studies but not Python.

You paste one prompt. The AI calls the `ratio_study` package for you through
**MCP** (Model Context Protocol) and hands back the numbers, a PDF report and the
Vertical Equity Panel.

---

## Before you start: will your Copilot actually do this?

This matters, and it is the step people skip. MCP lets an AI assistant call
software on your behalf — but not every Copilot surface can reach a locally
installed Python package.

| Where you are chatting | Can it call this package? | What IT has to do |
|---|---|---|
| **GitHub Copilot Chat in VS Code**, *Agent* mode | **Yes** — this is the easiest route | Install the package and drop in one config file (below) |
| **A Copilot Studio agent**, surfaced in Microsoft 365 Copilot | **Yes** | Host the MCP server on a reachable HTTPS endpoint and register it as a tool in Copilot Studio |
| **Microsoft 365 Copilot chat**, out of the box | **No** | It runs in the cloud and cannot see software on your laptop. Use one of the two routes above |
| **Claude Desktop / Cursor / Continue.dev** | **Yes** | Same one config file |

If you are not sure which you have, paste the **Setup check** prompt below into
your Copilot and read what it says.

> If none of these are available to you, the package can still be driven by
> Copilot as a *coach* — it writes the Python and tells you where to paste it.
> See [MCP_COPILOT_365_GUIDE.md](MCP_COPILOT_365_GUIDE.md) for that route.

---

## What IT needs to do (once, five minutes)

```bash
python -m venv ~/ratio_study_env/.venv
source ~/ratio_study_env/.venv/bin/activate        # Windows: ~\ratio_study_env\.venv\Scripts\activate
pip install "ratio_study[spatial,ml,interactive]" mcp
```

Then create `.vscode/mcp.json` in the folder you work in (VS Code), or add the
same block to your Claude Desktop config:

```json
{
  "servers": {
    "ratio_study": {
      "type": "stdio",
      "command": "/Users/YOU/ratio_study_env/.venv/bin/python",
      "args": ["-m", "ratio_study.mcp_server"]
    }
  }
}
```

Replace the `command` path with the venv Python that IT created. On Windows it
ends `\.venv\Scripts\python.exe`.

---

## Setup check

Paste this first. It confirms the connection before you rely on it.

```text
Using the ratio_study MCP server, call diagnose_setup and
get_module_capabilities. Tell me in plain language:

1. Is the ratio_study package installed, and which version?
2. Which optional extras are available (spatial, ml, interactive)?
3. Which tools can you call on my behalf?
4. Is anything missing that I should ask IT to install?

Do not write any Python for me to run — use the tools directly.
```

You should see seven tools: `get_module_capabilities`, `diagnose_setup`,
`list_datasets`, `inspect_dataset`, `run_ratio_study`, `generate_ratio_report`
and `generate_vertical_equity_panel`. If you see none, MCP is not connected —
go back to the table at the top.

---

## The main prompt

Fill in the four bracketed lines and paste the whole thing.

```text
You are helping me run a ratio study. I know assessment and the IAAO Standard
on Ratio Studies (September 2026); I do not write Python. Use the ratio_study
MCP tools to do the work — never ask me to run code myself.

MY DATA
- File: [C:\path\to\my_sales.csv]
- Sale price column (time-adjusted): [time_adjusted_sale_price]
- Assessed value column: [assessment_value_new]
- Stratify by: [neighbourhood]        (or write "none")
- Write outputs to: [C:\path\to\output_folder]

DO THIS, IN ORDER

1. Call inspect_dataset on my file. Show me the row count and the columns you
   found. If the column names I gave you do not exist, stop and show me the
   closest matches rather than guessing.

2. Call run_ratio_study with those columns, stratified as I asked, with
   indicators Median, CoD, PRD, PRB, VEI, VEM and ci_conf_level 90.
   Then tell me, in plain language and referencing the Standard:
   - the overall level of valuation and whether it sits in 0.90-1.10 (s.11.1);
   - the COD against the 5.0-15.0 range for residential improved
     (Table 11.1) - and warn me if it is BELOW the range, because s.11.2.5
     treats that as a sales-chasing / overfitting signal, not a good result;
   - the VEI against the -10% to +10% range (s.11.2.6), remembering that
     negative is regressive;
   - which strata fall outside the standards, and which are too small
     to judge.
   For every measure, tell me whether the confidence interval CONFIRMS the
   breach. A point estimate on its own is an indication, not proof (s.7).

3. Call generate_vertical_equity_panel writing to my output folder, and
   summarise the four gauges (PRB, PRD, VEI, VEM) and what they agree or
   disagree about.

4. Call generate_ratio_report writing to my output folder, with
   descriptive comments on, export_ratios "xlsx", the panel included, and
   these opening comments:
   - the purpose of the study,
   - the sales period and that prices are time-adjusted,
   - the standards being applied.

5. Finish with a short summary I could paste into an email: what passed,
   what failed, and what I should look at next.

RULES
- Use the MCP tools. Do not write Python for me to copy.
- If a tool returns an error, show me the message and what you will try next.
- Do not state a conclusion the confidence interval does not support.
- Ratio study statistics describe a population. Never use them to comment on
  an individual property (s.3.1).
```

---

## Follow-up prompts

Short prompts for once the first run is done.

**Check for sales chasing**
```text
My COD came out low. Using the ratio_study package, screen my file for sales
chasing with the Appendix B techniques and explain whether the low COD is
trustworthy. Tell me which tests you could NOT run and what column would
enable them.
```

**Trim outliers and show me the difference**
```text
Re-run the study with a 3 x IQR outlier trim computed within each
neighbourhood, and show me the before-and-after side by side. Tell me what
percentage of sales was trimmed and whether that is within the 5-10%
Appendix C.5 allows. Save the trimmed records so I can review them.
```

**Compare two models**
```text
Run the study twice on the same file - once using [assessment_value_old] and
once using [assessment_value_new] - and compare them. Which indicators
improved, which got worse, and which strata drove the change?
```

**Tighter confidence**
```text
Re-run the overall study at a 95% confidence level with a fixed seed of 7 so
the numbers are reproducible, and tell me which conclusions change.
```

**Explain a result to a non-technical audience**
```text
Write three short paragraphs for a council report explaining what the VEI of
[value] means, why it matters to taxpayers, and what we are doing about it.
No jargon, no statistics notation.
```

---

## Things to hold the AI to

Ratio studies are easy to over-read, and an eager assistant will over-read
them. Push back if you see any of these:

| If the AI says… | Ask it… |
|---|---|
| "Your assessments are 4% too low" | Does the confidence interval exclude 1.00, or is that just the point estimate? |
| "The COD of 3% shows excellent uniformity" | Is that below the Table 11.1 range? Have you screened for sales chasing? |
| "This property is over-assessed" | Ratio study statistics describe a population, not a property (s.3.1). |
| "The PRD shows no regressivity" | The PRD is being retired (s.11.2.6). What does the VEI say? |
| "Neighbourhood X fails" (12 sales) | Is that stratum large enough to conclude anything (s.9)? |
| It wrote Python for you to run | You have MCP tools. Use them. |

---

## If something goes wrong

```text
Something is not working. Call diagnose_setup and tell me in plain language
what is wrong and exactly what to send to IT. Do not try to fix my Python
installation yourself.
```

Common causes: the `command` path in `mcp.json` points at the wrong Python; the
`mcp` package was not installed in the venv; the output folder does not exist
(the tools will not create it); or the file is over the server's size limit.

---

*See also: [MCP server reference](mcp-server.md) · [Getting started](getting-started.md) ·
[API reference](reference/README.md)*
