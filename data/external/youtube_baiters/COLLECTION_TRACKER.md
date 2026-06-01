# YouTube Baiter Collection Tracker

Target by end of week: 50 scam calls and 30 legitimate calls.

## Current Counts

- Candidate pool: 50 scam + 30 legit in `candidate_pool_50_30.tsv`
- Collection list: 80 clips in `video_list_50_30.tsv`
- Transcribed + labeled annotations: 50 scam + 30 legit in `annotations.tsv`
- Timestamp status: 12 transcript-reviewed windows, 68 candidate windows pending transcript cleanup

## Workflow

1. Add candidate clips to `video_list.tsv`.
   - For the 50/30 collection, use `video_list_50_30.tsv`.
2. Run transcription:

   ```bash
   source .venv/bin/activate
   python experiments/collect_youtube_scambaiters.py \
       --video-list data/external/youtube_baiters/video_list_50_30.tsv \
       --output-dir data/external/youtube_baiters \
       --whisper-model tiny.en
   ```

3. Fill `annotations.tsv` labels after reviewing transcripts.
4. Verify loader counts:

   ```bash
   python experiments/dataset_loader.py
   ```

5. Re-run paper experiments:

   ```bash
   python experiments/run_all.py --sources repo_real youtube_baiters
   python experiments/extract_paper_numbers.py
   ```

## Candidate Mix

- Use at least 3 scam-baiter channels.
- Use at least 4 scam categories.
- Avoid clips with music, commentary overlays, or mostly baiter speech.
- Legitimate calls can come from screened public non-scam call recordings or manually screened synthetic/owned transcripts, but keep them clearly marked as `label=0`.
