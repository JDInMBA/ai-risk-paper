"""12_submission_docs.py - cover letters, title pages and highlights for each target journal."""
import re, subprocess
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location("b", Path(__file__).with_name("11_build_manuscripts.py")); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
DR = b.DR
def to_docx(md, path):
    p = Path(path).with_suffix(".md"); p.write_text(md)
    subprocess.run(["pandoc", str(p), "-o", str(path), "--reference-doc", str(b.REFDOC)], check=True); p.unlink()
def counts(md_path):
    md = Path(md_path).read_text(); body = md.split("# References")[0]
    figs = len(re.findall(r"!\[Figure \d+", md)); tabs = len(set(re.findall(r"\*Table (\d+)\.", md)))
    body = re.sub(r"\|.*\|", " ", body); body = re.sub(r"!\[.*?\]\(.*?\)\{.*?\}", " ", body)
    return len(re.findall(r"\b\w[\w'.%-]*\b", body)), figs, tabs
LETTER = """[Your name]
[Department, Institution]
[City, Country]
[Email] | [Phone]

[Date]

The Editor{ed}
{journal}

Dear Editor,

Re: Submission of the manuscript "{title}"

I am submitting the above manuscript for consideration as {atype} in {journal}.

Frontier AI laboratories have committed to spend many times their revenue, and listed cloud, chip and data-centre suppliers now carry those commitments as backlog. Anthropic's 2026 registration statement puts its committed spend at about $895 billion against trailing revenue of $35 billion, and OpenAI has filed confidentially for its own listing. The paper asks whether this dependence shows up in suppliers' tail risk, whether standard VaR models cover it and whether machine learning can predict it, using daily data on 69 US-traded firms from 2021 to November 2026, a frontier-lab dependence score built from SEC filings, and the contract terms in the S-1.

The main findings are that lab-dependent firms became the riskiest part of the AI sector after September 2025 but that their extra risk runs through exposure to the AI factor rather than a separate crash channel; that standard VaR models under-cover a lab-dependent basket at the 95% level while passing for control firms; that machine-learning models cannot time these tail events; and that contract-level stress losses, though material for Oracle, CoreWeave and Cerebras, are far smaller than the drawdowns those firms have already suffered. {fit}

The manuscript is original, has not been published elsewhere and is not under consideration by another journal. I am the sole author and have approved the manuscript for submission. I declare no competing interests, financial or non-financial.

All data are from public sources, and the code and processed data that reproduce every table and figure are deposited in a public repository with a persistent identifier, cited in the Availability of data and materials statement. The analysis is deterministic: two independent runs reproduce every reported table bit for bit, and the package includes the script that checks this.

Thank you for considering the submission.

Yours sincerely,

{author}
"""
J = {
 "02_Financial_Innovation": dict(journal="Financial Innovation", ed="s-in-Chief", atype="a research article",
     fit="The paper combines financial-risk measurement with machine learning and addresses a financial innovation (contract-based financing of AI infrastructure and the listing of frontier AI labs) that is new to the literature, which fits the journal's scope.",
     md="Manuscript_Financial_Innovation.md", anon=False, short="FI"),
 "03_Journal_of_Risk": dict(journal="The Journal of Risk", ed="", atype="a research paper",
     fit="Its core contribution is a validation of market-risk models (Kupiec and Christoffersen tests, Basel traffic light, Diebold-Mariano comparisons and an expected-shortfall backtest) for a new and concentrated source of equity risk, which matches the journal's focus on risk measurement.",
     md="Manuscript_Journal_of_Risk_ANONYMISED.md", anon=True, short="JoR"),
 "04_IIMB_Management_Review": dict(journal="IIMB Management Review", ed="", atype="a research article",
     fit="The paper ends with practical recommendations for investors, underwriters, suppliers and regulators, including disclosure ratios and a note on relevance for Indian investors who hold these exposures through global funds.",
     md="Manuscript_IIMB_Management_Review_ANONYMISED.md", anon=True, short="IIMB"),
 "05_Vikalpa": dict(journal="Vikalpa: The Journal for Decision Makers", ed="", atype="a research article",
     fit="The executive summary and Section 8 translate the results into decisions for portfolio managers, risk officers, underwriters and regulators, in keeping with the journal's decision-oriented readership.",
     md="Manuscript_Vikalpa_ANONYMISED.md", anon=True, short="Vikalpa"),
}
for folder, j in J.items():
    if not (DR / folder / j["md"]).exists():
        continue   # that variant has not been built
    od = DR / folder
    to_docx(LETTER.format(title=b.TITLE, author=b.AUTHOR, **{k: j[k] for k in ["journal","ed","atype","fit"]}), od / f"Cover_Letter_{j['short']}.docx")
    if j["anon"]:
        w, nf, nt = counts(od / j["md"])
        tp = f"""# Title page

Title: {b.TITLE}

Running header: {b.RUNNING} ({len(b.RUNNING)} characters)

Author(s): [Author Name]^1^

^1^ [Department, Institution, City, Country]

Corresponding author: [Author Name], [postal address], [email], [phone]. ORCID: [0000-0000-0000-0000]

Word count (main text, excluding tables and references): {w:,}

Number of figures: {nf}. Number of tables: {nt} in the main text plus appendix tables.

Keywords: {'; '.join(b.KEYWORDS)}

JEL classification: {b.JEL}

Funding: [State funding sources, or "This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors."]

Declaration of competing interest: The author(s) declare no competing interests.

Data availability: All data are from public sources (Yahoo Finance, FRED, SEC EDGAR). Code and processed data are available from the corresponding author and will be deposited in a public repository on acceptance.

Acknowledgements: [Optional]
"""
        to_docx(tp, od / f"Title_Page_{j['short']}.docx")
        print(folder, "words", w, "figs", nf, "tables", nt)
if (DR / "04_IIMB_Management_Review").exists():
    hl = "# Highlights\n\n" + "\n".join(f"- {h}" for h in b.HIGHLIGHTS) + "\n"
    to_docx(hl, DR / "04_IIMB_Management_Review" / "Highlights_IIMB.docx")
w, nf, nt = counts(DR / "02_Financial_Innovation" / "Manuscript_Financial_Innovation.md"); print("FI words", w, nf, nt)
