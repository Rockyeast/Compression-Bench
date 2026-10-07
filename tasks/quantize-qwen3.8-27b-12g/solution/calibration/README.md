# Fixed reference calibration data

`train.parquet` is the **train** split of Salesforce/wikitext,
`wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`.

Source: https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/train-00000-of-00001.parquet

Dataset license: CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/).
Original text: Wikipedia contributors, distributed via WikiText by Salesforce.
The Parquet file is unmodified.

This exact file was used in `ref12-gptq-20260906-01/run-01`.
The solution selects 128 disjoint 2048-token blocks with
seed 42. Calibration uses neither the public validation nor the hidden test split.
This directory belongs to the private oracle solution. The same training data
and sampling example are also provided to the Agent in /opt/calibration;
the reference quantization implementation itself remains private.
