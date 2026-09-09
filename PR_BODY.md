Title: Add Refresh Opportunity Scoring capstone paper and artifacts

Description:
This PR adds the full capstone deliverable: a reproducible Refresh / Content Opportunity Scoring pipeline built on the FlyRank internship dataset, plus a deployed research paper (served from `docs/`) and evaluation artifacts.

What this PR contains:
- `docs/index.html` — deployed research paper (HTML) with Abstract, Methodology, Results, and Action Playbook.
- `submission/paper_url.txt` — public URL placeholder for GitHub Pages.
- `scripts/01_prepare_features.py` & `scripts/05_train_refresh_model.py` & `scripts/06_eval_and_chart.py` — reproducible pipeline to prepare data, train a model, and generate evaluation charts.
- `outputs/` — model, metrics, top-ranked CSV, and charts (generated locally).
- `work/notebooks/w08_paper_publish.ipynb` — final reproducible notebook summarizing results and reproducing figures.
- `.github/workflows/deploy_docs.yml` — workflow to publish `docs/` to the `gh-pages` branch on push to `main`.

Checklist (before merging):
- [ ] Confirm `submission/paper_url.txt` contains the final GitHub Pages URL (update after enabling Pages or after first successful workflow run).
- [ ] (Optional) Review `outputs/top_ranked_refresh.csv` to confirm no private or sensitive content is included.
- [ ] Run the one-command reproducibility line locally and verify the charts are present in `outputs/charts/`.
- [ ] Enable GitHub Pages in repository settings if needed (the workflow will create `gh-pages` branch automatically). If Pages is already enabled to serve from `gh-pages` or `docs/`, the site should appear at the URL in `submission/paper_url.txt` after the action runs.

Notes on Pages & deployment:
- This workflow publishes `docs/` to the `gh-pages` branch using the repository's `GITHUB_TOKEN`. On the first successful run it will create `gh-pages` and commit the site contents; GitHub Pages may take a minute to publish the site.
- If your repo uses a branch protection policy that prevents the action from creating `gh-pages`, manually create the branch or allow the action to push.

Short summary for reviewers:
This PR implements the full capstone deliverable: reproducible pipeline, trained model, evaluation figures, and a public paper page. The results and methodology are documented in the included notebooks and `docs/index.html`.
