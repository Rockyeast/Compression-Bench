# Fixed evaluation inputs

`ppl_inputs.json` stores the fixed C4 token blocks. `ppl_config.json` defines the
tokenizer, split, token counts and BF16/Reference PPL baselines.
The evaluator checks the protocol, split, token counts and token-ID range before measuring PPL.

`decode_inputs.json` stores the fixed decode prompts. `decode_config.json`
 defines the request lengths, warmup, measurement rounds and scoring inputs. The evaluator uses the GPU allocated for that run.

Public and hidden evaluation use separate inputs. Hidden inputs belong only in
the Verifier environment. The Agent receives the task's public interface.

The public Checker's `evaluation_profiles.json` selects `ppl_assets`, `ppl_split`
and `decode_assets` for each supported profile. The optional 9B profile has its
own tokenizer-specific inputs and returns diagnostic measurements.
