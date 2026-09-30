# MCP server

`ratio_study` ships an [MCP](https://modelcontextprotocol.io) server, so an AI
assistant can run a ratio study directly instead of writing code for you to
paste and run.

```bash
pip install "ratio_study[spatial,ml,interactive]" mcp
python -m ratio_study.mcp_server        # stdio transport
```

For a copy-paste prompt aimed at non-programmers, see
[the Copilot prompt template](copilot-prompt-template.md).

---

## Client configuration

**VS Code** (`.vscode/mcp.json`), **Claude Desktop**
(`claude_desktop_config.json`), Cursor and Continue.dev all take the same shape:

```json
{
  "servers": {
    "ratio_study": {
      "type": "stdio",
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "ratio_study.mcp_server"]
    }
  }
}
```

Point `command` at the interpreter of the environment the package is installed
in — not a bare `python`, which may resolve elsewhere. Claude Desktop nests the
same block under `"mcpServers"` rather than `"servers"`.

---

## Tools

| Tool | Purpose |
|---|---|
| `get_module_capabilities` | Version, supported file types, available indicators, tool list |
| `diagnose_setup` | Whether the package and its extras import cleanly — start here when something fails |
| `list_datasets` | Readable datasets in a directory |
| `inspect_dataset` | Columns, dtypes and row count, so the assistant picks real column names |
| `run_ratio_study` | The study itself — indicators, stratification, trimming, confidence intervals |
| `generate_ratio_report` | A PDF report, optionally with XLSX/CSV alongside |
| `generate_vertical_equity_panel` | The Vertical Equity Panel as a PNG, plus the gauge values as text |

### Supported input formats

`.csv`, `.tsv`, `.parquet`, `.xlsx`, `.xls`, `.json`, `.sav` (SPSS), and the
compressed forms `.csv.gz`, `.parquet.gz`, `.csv.zip`, `.parquet.zip`. Files
above the server's size limit are refused rather than silently truncated;
`get_module_capabilities` reports the limit.

---

## `run_ratio_study`

Wraps [`ratios`](reference/ratio-study.md#ratios). Required: `file_path`,
`market_value_col`, `assessment_col`.

Useful optional arguments: `group_by`, `indicators`, `iqr_trim_multiple`,
`iqr_group_by`, `ci_mode` (`"bootstrap"` or `"distribution"`), `ci_conf_level`,
`ci_indicators`, `ci_seed`, `weights_col`, `return_expanded_dict`,
`mkt_value_proxy_col` (market-value proxy for VEI), `prb_mvp_col`
(market-value proxy for the PRB regression, overriding the blend it builds
itself — independent of `mkt_value_proxy_col`), `mvp_sale_price_share`
(share of the sale price in that blend; `0.5` is the Standard's 50/50 proxy,
`0.3` a 30/70 one).

Returns the formatted ratio table, a headline row, and the columns it used.
Plot objects and raw data are stripped from the response — the assistant needs
the numbers, not megabytes of figures.

## `generate_ratio_report`

Wraps [`ratios_report`](reference/ratio-study.md#ratios-report). Additionally
requires `output_directory`, which **must already exist**. Takes `model_name`,
`base_date`, `group_by`, `descriptive`, `export_ratios`
(`""`/`"csv"`/`"xlsx"`/`"both"`), `opening_comments`, `closing_comments`,
`include_vertical_equity_panel`, `iqr_trim_multiple`, `ci_conf_level`,
`ci_seed`, `mkt_value_proxy_col` (market-value proxy for VEI), `prb_mvp_col`
(market-value proxy for the PRB regression, independent of the other one).

Returns the filenames it wrote and the headline ratio row.

## `generate_vertical_equity_panel`

Wraps [`vertical_equity_panel`](reference/charts.md#vertical-equity-panel).
Requires `output_directory`; takes `filename`, `title`, `subtitle`,
`vei_conf_level`, `vem_conf_level`, `vem_acceptable_inequity_level`,
`mkt_value_proxy_col` (market-value proxy for VEI), `prb_mvp_col`
(market-value proxy for the PRB regression this tool recomputes
internally, since there is no precomputed PRB to reuse here).

Returns the PNG path **and** the four gauge values as text, so the assistant can
describe the verdict without opening the image.

---

## Notes and limits

- **Read-only on inputs.** The server never writes to your source data. The two
  generator tools write only into the `output_directory` you name, and will not
  create it for you.
- **Errors come back as data.** A failure returns `{"error": ..., "traceback":
  ...}` rather than raising, so the assistant can read the message and correct
  itself. If it keeps failing, ask it to call `diagnose_setup`.
- **The assistant chooses the arguments.** It can pick a wrong column or a wrong
  stratification as easily as a person can. Ask it to call `inspect_dataset`
  first and to state which columns it used — the tools return that in every
  response for exactly this reason.
- **Statistical judgement is still yours.** The tools compute; they do not
  decide whether a sample is representative, whether sales were properly
  screened, or whether a stratum is large enough to conclude anything.
