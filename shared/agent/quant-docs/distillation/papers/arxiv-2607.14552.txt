                                         Answer-Conditioned Chains of Thought Degrade Verifiable-Reasoning Distillation
                                                                 in Large Language Models
                                                      Jungseob Lee1 , Seungyoon Lee1 , Suhyune Son1 , Dongyub Jude Lee2 , Sungbin Han1 ,
                                                                              Sugyeong Eo3∗ , Heuiseok Lim1∗
                                                                                         1
                                                                                           Korea University
                                                                                                    2
                                                                                       Zoom Communications
                                                                                         3
                                                                                           Yonsei University
                                            {omanma1928, dltmddbs100, ssh5131, sungbinhan9039, limhseok}@korea.ac.kr, jude.lee@zoom.us, s.eo@yonsei.ac.kr

                                                                        Abstract                               answer, so the correctness filter keeps them.
                                           A standard recipe for distilling the reasoning ability of large         The fix rests on the implicit assumption that a correct
                                           language models (LLMs) is to sample chains of thought from          chain is a good training target no matter how it was produced
                                           the model, keep those that reach the correct final answer,          (Yuan et al. 2023; Singh et al. 2023; Xiong et al. 2025; Li and
                                           and fine-tune on the survivors. When sampling fails, a com-         Zhang 2026). We show that this assumption fails. A chain that
                                           mon fix shows the generator the gold answer and asks it to          reaches the right answer can still teach the wrong behavior
                                           write a chain that reaches that answer. We show that this sec-      when the model wrote it while looking at the answer, and the
arXiv:2607.14552v1 [cs.CL] 16 Jul 2026




                                           ond step degrades the training data in a way that correctness       standard correctness filter cannot catch the problem, because
                                           filtering cannot catch. We run a controlled experiment that         the chain is correct by construction.
                                           fixes the generator, the problem set, and the correctness filter,
                                           and varies only whether the chain is generated under answer-
                                                                                                                   To isolate the effect we run what we call the one-bit ex-
                                           conditioning, the gold answer shown with a request to reach         periment, summarized in Figure 1. Using a single model, we
                                           it. Training a strong instruction-tuned reasoning model on its      generate chains of thought for the same set of problems under
                                           own answer-conditioned chains sharply lowers its verifiable-        two conditions that differ by a single intervention, whether
                                           reasoning accuracy. The loss grows with difficulty, reaching        generation is answer-conditioned, the gold answer shown in
                                           as much as about 27 points on the hardest competition prob-         the prompt with a request to reach it. We keep only cor-
                                           lems. The mechanism is legible in the chains themselves,            rect chains in both conditions and fine-tune with an identical
                                           which rationalize backward from the shown answer instead of         recipe. Training on the answer-conditioned chains is sharply
                                           deriving it, with the early final-answer statement as the mea-      worse than training on the answer-blind ones. The drop per-
                                           surable symptom. The harm is a property of the data rather          sists under a length-matched control and a budget sweep and
                                           than the generator, read off unlabeled generations before any
                                           fine-tuning, ordering the penalty across eight thinking models
                                                                                                               grows with difficulty. Its size depends on the model, destroy-
                                           from four families, and transferring across teacher families.       ing accuracy on an already-strong one and forfeiting part of
                                           A prompt ablation localizes it to the rationalize-toward in-        a real answer-blind gain on one with room to improve.
                                           struction rather than the answer’s bare visibility. The practical       Throughout, answer leakage is our informal shorthand
                                           takeaway is to generate answer-blind, because no correctness        for this recipe, the move that deployed self-improvement
                                           filter can see this damage in the data. Code is available at        pipelines make when sampling fails (Zelikman et al. 2022;
                                           https://github.com/js-lee-AI/answer-leakage.                        Zhang et al. 2023; Yu et al. 2024; Li et al. 2025). The recipe
                                                                                                               bundles two ingredients, the answer’s visibility and the in-
                                                                1   Introduction                               struction to rationalize toward it. Our mechanism analysis
                                         Large language models reason through chains of thought                separates them, and the instruction is what carries the harm.
                                         (Wei et al. 2022; OpenAI 2024), and much recent progress              At fixed visibility a neutral instruction recovers most of the
                                         in open reasoning models comes from distilling this ability by        penalty.
                                         supervised fine-tuning on the models’ own chains of thought               A chain composed toward a known answer rationalizes
                                         (DeepSeek-AI et al. 2025). A typical pipeline samples candi-          instead of deriving, and its symptom is stating the answer
                                         date chains, keeps the ones that reach the correct final answer,      near the start. Within a single corpus, we find that these
                                         and fine-tunes the model on the survivors. When sampling              rationalized chains carry most of the harm, and that a simple
                                         fails to produce a correct chain, or when the available chains        filter dropping them is a partial repair. The same tendency
                                         look noisy, a popular fix is to condition the generation on           is measurable before any fine-tuning, where it predicts the
                                         the gold answer. The self-taught reasoner method feeds the            penalty across eight models from four families, transfers on
                                         answer as a hint and asks the model to rationalize problems           average when answer-conditioned chains from one model
                                         it could not solve (Zelikman et al. 2022), and answer-aware           train a smaller one. We study verifiable-reasoning distillation,
                                         refinement pipelines regenerate chains directly from question         domains such as math and code where a final-answer check
                                         and answer pairs (Zhang et al. 2023; Yu et al. 2024; Li et al.        or test execution lets the correctness filter run.
                                         2024, 2025). Chains produced this way still reach the correct             The effect is large where a derivation is required and ab-
                                            ∗                                                                  sent where it is not. For Qwen3-8B, the answer-conditioned
                                                Corresponding author.
                           Shared Generator     Prompt                        Generated Chain (CoT)             Final answer Filter       Same SFT              Student


           Standard
                                                                                          Chain c𝑖
       answer-blind path
                                                                                                           𝑎𝑖
         gold hidden                           Question 𝑞𝑖
                                                                                                                  keep chains whose           SFT on
                             Shared LLM 𝐺                                                                            final answer        filtered chains      Blind student
           from 𝐺
                               same model     answer hidden                                                           is correct        same init + recipe



                                                         only answer visibility differs



        Answer-visible                                                                    Chain c𝑖
        rationalization
                                                                       𝑎𝑖
          gold visible       Shared LLM 𝐺     Question 𝑞𝑖                                                         keep chains whose            SFT on         Rationalized
                                                                                                                     final answer         filtered chains       student
             to 𝐺              same model     gold answer 𝑎𝑖                                                          is correct         same init + recipe




Figure 1: The one-bit experiment. A shared generator produces chains for the same problems under an answer-blind path and
an answer-visible rationalization path. Both arms use the same final-answer filter and the same SFT recipe, so the arms differ
only in whether generation is answer-conditioned, the gold answer shown together with a request to reach it.


corpus trains a student 16.2 points weaker on MATH-500                                               The corrective costs nothing extra, since generating answer-
than the answer-blind corpus built from the same problems,                                           blind with the same correctness filter avoids the pathology
and the gap grows to 27.2 points on olympiad AIME prob-                                              and modestly beats distillation from gold data at a controlled
lems. Repeating the intervention in code degrades the student                                        budget.
as well, even though every kept chain passes its tests. On
multiple-choice knowledge and science benchmarks, which                                                                      2        Related Work
ask the model to select an option rather than write a deriva-
tion, we find no penalty, so the harm is confined to tasks                                           Rationalization in self-improvement pipelines. A promi-
whose answers must be derived.                                                                       nent line of self-improvement methods treats final-answer
   Our contributions are the following.                                                              correctness as the quality criterion for a chain (Xie et al.
                                                                                                     2025). The self-taught reasoner approach keeps the ratio-
1. We isolate the cost of answer-conditioned generation.                                             nales that reach correct answers and rationalizes the rest by
   Holding the generator, problems, correctness filter, data                                         feeding the model the answer as a hint (Zelikman et al. 2022),
   volume, and chain length fixed, we show that generat-                                             a hindsight relabeling of failed attempts (Zhang et al. 2023).
   ing chains under answer-conditioning incurs the leakage                                           The same move extends to token-level reasoning (Zelikman
   penalty, we localize it by a prompt ablation to the instruc-                                      et al. 2024) and recurs in rejection-sampling and ReST-style
   tion to rationalize toward the answer rather than to merely                                       methods (Yuan et al. 2023; Gulcehre et al. 2023; Singh et al.
   seeing it, and we replicate it on further models.                                                 2023; Zhang et al. 2024; Luong et al. 2024; Zhu et al. 2026)
2. We trace the mechanism. We find that a visible answer                                             and in answer-augmentation pipelines such as MetaMath (Yu
   moves chains toward stating the answer early, an off-                                             et al. 2024).
   policy shift away from the generator’s own search distri-                                            A closely related analysis, TRICE, shows that condition-
   bution, and we show that the habit transfers to the student’s                                     ing a chain on the answer biases the learning signal and
   own outputs even with answer correctness controlled.                                              recasts the objective as latent-variable inference to correct
3. We attribute the harm to the rationalized chains and                                              it (Phan et al. 2023). We instead show that this bias is in-
   partly repair it with a filter. In an experiment inside a                                         visible to the standard correctness filter, causally harms su-
   single answer-conditioned corpus, we show that these                                              pervised fine-tuning even when every chain is verified, and
   chains carry most of the harm, a three-seed difference-                                           can be anticipated from unlabeled generations before any
   in-differences of 16.9 points.                                                                    training. A different strategy trains a verifier on both correct
4. We predict the penalty before any training. We read a sig-                                        and incorrect chains rather than filtering them away (Hos-
   nature off unlabeled generations that orders the leakage                                          seini et al. 2024). Our results show that how a correct chain
   penalty across eight models from four families and pre-                                           was produced still matters. To our knowledge, ours is the
   dicts a held-out family with a mean absolute error of 2.9                                         first controlled isolation of answer-conditioning, holding the
   points.                                                                                           generator, problems, correctness filter, and recipe fixed and
                                                                                                     varying only whether the generator is shown the answer.
   These findings come with cheap safeguards. A candidate
teacher can be screened before any student is trained, since                                         Backward reasoning without leakage. Reverse-
the signature is measured on a small set of its unlabeled                                            reasoning methods that generate backward questions or
generations, and where a pipeline cannot avoid showing the                                           check forward and backward consistency without revealing
gold answer, an instruction to derive first and only then state                                      the gold answer are leakage-free and are not implicated by
the answer restores about two-thirds of the lost accuracy.                                           our findings (Chen et al. 2024; Kazemi et al. 2023; Weng
et al. 2023). We study the answer-visible regime that those           Setting / Benchmark               Blind ↑      Leaked ↑   Penalty ↓
methods avoid.
                                                                      (A) Instruction ablation at fixed visibility
Process vs. outcome supervision. Outcome-supervised                   Rationalize toward answer          95.1          78.9         16.2
                                                                      Derive-first                       95.1          90.7          4.3
verifiers (Cobbe et al. 2021), comparisons of process and
                                                                      Final-check                        95.1          93.7          1.3
outcome feedback (Uesato et al. 2022), and process reward
                                                                      Out-of-band                        95.1          94.3          0.7
models (Lightman et al. 2023; Wang et al. 2024; Luo et al.
2024; Zheng et al. 2025; Song et al. 2025; Zhang et al.               (B) Within-corpus chain-population carrier
2025; Wang et al. 2026) all argue that final-answer correct-          Leaked-Early                   94.0        64.2               29.8
ness is an insufficient quality signal and supervise or rerank        Leaked-Late                    94.8        86.2                8.6
individual steps. We show a complementary failure on the              Difference-in-differences                                   +16.9
data-generation side, where chains that are correct and step-
wise plausible can still harm the student because of how they         (C) Across math benchmarks, difficulty-ordered
were generated.                                                       GSM8K (grade school)           95.4       90.5                4.9
                                                                      MATH-500 (competition)         95.1       78.9               16.2
Faithfulness of chain-of-thought. Chains of thought can                 Level 1                      95.3       95.3                0.0
be unfaithful or post-hoc. They are unfaithful when a model             Level 2                      98.1       84.1               14.1
exploits a hint injected at inference time without verbalizing          Level 3                      98.4       83.5               14.9
it (Turpin et al. 2023; Chen et al. 2025; Tutek et al. 2025; Xu,        Level 4                      95.6       74.2               21.4
Qi, and Xu 2024), and post-hoc when the stated reasoning                Level 5                      89.8       70.9               18.9
does not drive the final answer (Lanham et al. 2023; Paul             Minerva (graduate STEM)        48.7       38.5               10.2
et al. 2024). We show the training-time analogue, where               AIME 2024–2025 (olympiad)      71.7       44.4               27.2
hint-conditioned targets transfer the post-hoc habit to the
student, producing a measurable shift in answer position and       Table 1: Answer-leakage penalty on Qwen3-8B, MATH-500
an accuracy cost.                                                  accuracy unless a benchmark is named, with Penalty the
                                                                   Blind minus Leaked gap of Eq. (1). The tinted Blind column
Data selection and on-policy distillation. Data-selection          is the recommended answer-blind arm. (A) varies the gener-
work argues that fewer and better examples can outperform          ation instruction at fixed answer visibility. (B) is the within-
larger noisy sets (Zhou et al. 2023), but our harmful examples     corpus two-by-two carrier design. (C) spans four difficulty-
already pass the correctness criterion, so the failure mode        ordered math benchmarks.
is distinct from low quality. On-policy distillation reports
advantages for self-generated data under a soft-KL objective
(Agarwal et al. 2024; Sang et al. 2026), which is consistent       is built on a fixed set of 935 math problems, each carry-
with our answer-blind result, but that setting changes the         ing a gold answer a(q). Holding all of this fixed, the lone
generator and the objective at the same time and never isolates    intervention is answer-conditioning.
answer visibility.                                                    Let M be the generator. For each problem we build two
                                                                   prompts. The Blind prompt (answer-blind) shows only the
Concurrent leakage filters. Concurrent work has begun              problem and asks the model to derive the answer. The
to patch leakage symptoms inside refinement pipelines, for         Leaked prompt (answer-leaked) appends a(q) and asks for a
example gating hindsight-hint repairs with an entropy filter       reasoning trace that arrives at it. From M we sample chains
(Zhang et al. 2026b,a; Li et al. 2026), but such mitigations do    under each prompt, keep only chains whose final boxed an-
not isolate what is being filtered or why. Our one-bit experi-     swer is correct, and fine-tune two students on the result-
ment supplies the missing causal quantity, the accuracy cost       ing (question → chain) pairs under one identical recipe, an
of answer-conditioned generation in an otherwise identical         answer-blind student Sblind and an answer-leaked student
pipeline, together with a test isolating which chains carry the    Sleaked . The reported leakage penalty is the accuracy lost by
penalty inside one corpus and a predictor read off unlabeled       generating the chain under answer-conditioning,
generations before any fine-tuning.
                                                                                  ∆ = acc(Sblind ) − acc(Sleaked ).                         (1)

           3    Isolating Answer Leakage                           The generator, problem set, correctness filter, recipe, and the
                                                                   matched examples are held fixed, so the two students differ
Unless noted otherwise, every trained model is a full-             only in answer-conditioning and in the chain-length shift it
parameter supervised fine-tune of Qwen3-8B (Qwen Team              induces, which we control with a length-matched arm below.
2025) under one fixed recipe, on training problems from               Table 1 shows that generating under answer-conditioning
the math split of the Nemotron-Cascade-SFT-Stage-2 cor-            costs 16.2 points on MATH-500 as a three-seed mean. Be-
pus (Wang et al. 2025), disjoint from all evaluation sets. We      cause every other factor is held fixed, this gap is the leak-
evaluate with vLLM (Kwon et al. 2023) in thinking mode             age penalty ∆ of Eq. (1). The Blind student sits at the
on MATH-500 (Hendrycks et al. 2021) with a 32,768-token            base model within seed noise (95.1 against 95.2), so answer-
budget and symbolic answer matching. Under this harness            blind self-distillation preserves performance and essentially
the Qwen3-8B base model scores 95.2. Appendix G gives the          the whole gap is leakage damage. Table 1(C) evaluates the
full recipe, sampling, and harness details. The comparison         same fine-tunes on four independent math benchmarks from
                                                  (a) Leaked chains state the answer first                         (b) The habit propagates to accuracy                           (c) Train/test length inverts
                                                                                                             100                              95.2
                                           1.75                                     Blind                                                                                    30                      Blind
                                                                                    Leaked                    90       Blind                                                                         Leaked
                                                                                                                                                 -15.6
                                           1.50                                                                        Leaked
                                                                                                              80




                                                                                                                                                         Length (1k chars)
                       Density of chains
                                                                                                                                                                             25
                                           1.25                                                                                               79.6
                                                                                                              70




                                                                                                   Percent
                                           1.00                                                               60                                                             20
                                                                                                                    47.4         48.7
                                           0.75                                                               50
                                                                                                                                    +12.2
                                           0.50                                                               40                                                             15
                                                                                                                        +20.8
                                                                                                              30                 36.5
                                           0.25
                                                                                                              20                                                             10
                                                                                                                    26.6
                                           0.00
                                               0.0        0.2    0.4        0.6   0.8        1.0                   Training     Student     MATH-500                                 Train           Test
                                                                                                                     data        chain      accuracy                                chains          chains
                                                      First-answer position (fraction)                               AFR          AFR



Figure 2: Answer-position and length behavior of the answer-blind and answer-leaked arms in training chains and student
outputs. (a) Distribution of the first gold-answer position within the think block for the answer-blind chains, generated with
the answer never shown, and the answer-leaked chains, generated with the gold answer shown. (b) Answer-first rate (AFR,
Eq. (2)) in the training corpus and in the trained students’ own answer-blind generations, for the blind and leaked arms. (c) Mean
think-block length of the blind and leaked arms in the training chains and in the students’ own test generations.


                                                                Qwen3-8B          Qwen3-4B                                        We read it off the matched pairs through answer position and
                                                                                                                                  the student’s test-time behavior, account for it as an off-policy
    Leakage penalty




                      30
                                                                                                                                  shift in the generation distribution, and then ask whether the
                      20
                                                                                                                                  same shift identifies which chains inside a single corpus carry
                                                                                                                                  the harm.
                      10
                                                                                                                                  4.1       How the leakage penalty manifests
                       0                                                                                                          A visible answer makes the model state the answer early.
                                           4          8                16               24                    32
                                                                                                                                  This reading rests on a single chain-level property. A chain
                                                           Evaluation token budget (K)
                                                                                                                                  is answer-first if the gold final answer is stated within the
                                                                                                                                  first 20% of the think block, and derivation-first otherwise.
Figure 3: Leakage penalty against the evaluation token bud-                                                                       Writing pos(c) ∈ [0, 1] for the relative position of the first
get, scoring a problem correct only when its boxed answer                                                                         gold-answer mention inside the reasoning of a chain c, the
completes within the budget.                                                                                                      answer-first rate of a corpus C is
                                                                                                                                                            1 X
                                                                                                                                             AFR(C) =               1[pos(c) < 0.2] .       (2)
grade-school GSM8K to olympiad AIME, and the penalty                                                                                                       |C|
                                                                                                                                                                                   c∈C
holds across all four. Within MATH-500 it grows with prob-
lem difficulty, from near zero on the easiest problems to over                                                                    A high AFR is the measurable symptom of rationalizing
twenty points on the hardest levels. On the pooled Level 4–5                                                                      toward a known answer rather than deriving it. Figure 2a
band the penalty is 20.1 points with real headroom in the                                                                         shows the leaked chains landing in the answer-first zone far
blind arm, with the band-level and seed-level numbers in                                                                          more often than blind, with the whole position distribution
Appendix B. Appendix G reports the seed-level scores and                                                                          shifted toward the start of the think block. On the matched
Appendix I the full inferential statistics.                                                                                       pairs this is a 20.8-point rise in AFR at comparable answer
                                                                                                                                  coverage, detailed in Appendix C.
The penalty is content, not length. Appendix G reports a                                                                             Table 1A holds the answer visible across all arms and
length-matched control that rules out length as the explana-                                                                      varies only the instruction. The penalty falls with the answer-
tion. To check that length is not the cause, we truncate each                                                                     first rate to under a point, so the harm tracks the instruction,
blind chain, which runs longer, to its matched leaked chain’s                                                                     not the answer’s mere visibility. The derive-first instruction
length, preserving genuine answer-blind reasoning and the                                                                         shifts the answer-first rate to 36.0%, between the leaked
correct answer, and fine-tune on it. This length-matched arm                                                                      47.4% and the blind 26.6%, and recovers about two-thirds
scores 95.0, far above Leaked, so the penalty is carried by                                                                       of the penalty. The two neutral arms still land about a point
the answer-conditioned content, not the length.                                                                                   below blind, a residual cost of bare visibility set against the
                                                                                                                                  fifteen the instruction carries.
                                                  4       Mechanism Analysis                                                      Two deficits underlie the penalty. Figure 3 shows the
Answer-conditioning pushes the model to state the final an-                                                                       penalty growing as the evaluation budget shrinks, the mark
swer early and narrate toward it rather than derive it, a shift                                                                   of a termination deficit, and the damage splits into two mea-
measurable in unlabeled generations before any fine-tuning.                                                                       surable components. The derivation deficit is error when the
                                  Answer-first rate AFR           gone, exactly the early-answer narration we observe. Two
         Student outputs           All       Correct              questions follow, whether the damage travels with the ratio-
                                                                  nalized chains inside a single corpus and whether ∆AFR
         Base (no SFT)             38.7        37.1
         Blind (answer blind)      36.5        35.8
                                                                  flags the harmed models before training.
         Leaked (answer leaked)    48.7        45.2
                                                                  4.3      Isolating the rationalized chains
         AFR leaked − blind       +12.2       +9.4
                                                                  The first open question is whether the harm concentrates in
                                                                  the rationalized chains. Inside the single leaked corpus the
Table 2: Answer-first rate (AFR, Eq. (2)) of each student’s       mechanism account expects it to, with a chain-level filter
own MATH-500 generations under the answer-blind solve             that drops those chains acting as a drop-in fix. We test this
prompt. The tinted Blind row is the recommended answer-           causally.
blind arm.
                                                                  A within-corpus split isolates the rationalized chains.
                                                                  We split the 935 leaked chains by whether the answer is
student does finish, on problems where both students pro-         stated within the first 20% of the think block, giving Leaked-
duce an extractable answer the leaked one is still 9.5 points     Early (438 chains) and Leaked-Late (497), fine-tune sep-
worse on Qwen3-8B and 21.1 on Qwen3-4B. The termi-                arately on each with the same recipe, and read the four arms
nation deficit is failure to finish at all, the leaked student    in Table 1(B).
producing no extractable answer within the budget on 7.4 of          The split conditions on a leaked-chain property, so it could
every hundred problems against 0.4 for blind on Qwen3-8B,         track problem difficulty, and the blind twins of the two sub-
and 12.6 against 1.8 on Qwen3-4B. We trace this trained-in        sets confirm that it partly does, as Appendix F quantifies.
pathology to a train/test length inversion.                       We therefore add two control arms on the blind chains of
                                                                  the same problems, Blind-PE and Blind-PL, turning the
Leaked chains invert the train/test length relationship.          comparison into a difference-in-differences (DiD) that nets
Figure 2c shows the starkest behavioral fingerprint, a            out problem-subset effects. A length account predicts the op-
train/test length inversion. At training time the leaked chains   posite ordering here, because the rationalized chains are the
are shorter because seeing the answer compresses the rea-         longer ones, 21.4K against 18.3K characters. Appendix F
soning, yet at test time the leaked-trained model generates       confirms that the split is robust to the answer matcher, the
longer chains than the blind-trained one while scoring worse.     position threshold, and the handling of chains that never state
Trained on short answer-aware chains, it never learned to de-     the answer.
rive an unknown answer and generates at length without
converging, the termination deficit above.                        Primary endpoint. Reusing the 0.2 answer-first thresh-
                                                                  old from the signature above, the primary endpoint is the
4.2   The habit is off-policy rationalization                     difference-in-differences
The answer-first habit transfers to the student’s own out-                 DiD = (Leaked-Late − Leaked-Early)
                                                                                                                        (3)
puts. Table 2 shows that the tendency is not confined to the                        − (Blind-PL − Blind-PE),
training corpus, reappearing in each student’s own MATH-          evaluated over the 500 evaluation problems. Exact thresholds
500 generations where no answer is shown, so the leaked           in Appendix F mark three outcomes, a full carrier when
student has internalized the habit. Scoring AFR (Eq. (2))         Leaked-Late recovers to the blind arms, a partial carrier
on each student’s evaluation chains against each chain’s own      when it sits well above Leaked-Early but still below blind,
boxed answer, the leaked student is answer-first more often       and no carrier when it is not decisively positive.
than blind, which sits with the base. A correctness control
on the 387 problems all three students solve preserves a 9.4-     The rationalized arm carries most of the harm. Ta-
point gap, and the median first-answer position stays earlier     ble 1(B) reports the four arms and the primary endpoint.
for the leaked student. The transfer is partial, because leaked   The model trained on rationalized leaked chains collapses
chains run about 29% longer, as Appendix D details.               while the derivation-first arm largely holds. The two blind
                                                                  controls do not differ, so the easy-answer problems train per-
Answer-conditioned chains are off-policy rationaliza-             fectly well once the chains are blind. The three-seed mean of
tions, not search. We read these chains as rationalizations,      the endpoint is +16.9 (seed range 12 to 21).1
coherent narratives toward a known conclusion that satisfy           Against their own blind twins the rationalized arm loses
outcome filters while encoding a search-free policy whose         about 3.5× as much as the derivation-first arm. The ratio-
symptom is the high answer-first rate. The answer-blind gen-      nalized chains therefore carry most but not all of the harm,
erator samples from P (c | q), the search distribution a de-      and a chain-level filter is a cheap partial repair that leaves
ployed solver must follow because the answer is unknown           a residue only blind generation closes. We scope the causal
at test time. Answer-conditioning instead samples from the        claim to the chain population marked by the signature and
lower-entropy P (c | q, a(q)) and collapses that search to        sharpen it to the single answer-stating line when we probe
post-hoc justification of a known target. Hard supervised         generalization. Appendix I reports the full interval analysis
fine-tuning is on-policy only with P (c | q) targets. Targets     and Appendix F the robustness of the split.
drawn from P (c | q, a(q)) are therefore off-policy and teach
                                                                     1
a distribution the student cannot occupy once the answer is              A single-seed run gives +21.2.
                    Pre-FT signature          Post-FT outcome                                                                            Leakage Penalty
   Model            Base ↑ ∆AFR Blind ↑ Leaked ↑ Penalty ↓            Held-out model                                                   Predicted         Observed
   Held-in models
                                                                      Qwen3-1.7B                                                         24.4                29.8
   Qwen3-8B          95.2   +20.8      95.1       78.9      16.2
   Qwen3-4B          96.0   +26.6      94.0       64.8      29.2      DS-R1-Distill-Qwen-14B                                              7.5                 8.8
   GLM-Z1-9B         94.4   +11.1      94.8       82.6      12.2      Llama-Nemotron-8B (third family)                                   17.4                17.6
   DS-Qwen-7B        79.6   +11.0      89.7       82.4       7.3      GLM-Z1-9B                                                          10.1                12.2
   DS-Llama-8B       77.8   −0.2       65.8       64.6       1.2
   Held-out models                                                 Table 4: Held-out leakage penalties predicted from the
   Qwen3-1.7B      92.6     +26.2      89.6       59.8     29.8    answer-first signature against the penalties later observed.
   DS-Qwen-14B 88.2         +9.3       92.2       83.4     8.8     Each prediction was made before that model’s fine-tunes
   Nemotron-8B     95.4     +17.9      92.2       74.6     17.6    were evaluated. Llama-Nemotron-8B is an independent third
                                                                   family.
Table 3: The answer-first signature across eight thinking
models from four families, each self-distilled blind against
                                                                                                                                                         Qwen3-1.7B
leaked on MATH-500. Base is un-fine-tuned accuracy,                                      30   linear fit
                                                                                              in-sample
∆AFR the pre-fine-tuning signature of Eq. (2). The Qwen3-                                25   held-out (predicted)
                                                                                                                                                         Qwen3-4B




                                                                       Leakage penalty
8B and DS-Qwen-7B rows are three-seed means, the other                                   20                                            Llama-Nemotron-8B
six single training runs.                                                                15                                                  Qwen3-8B
                                                                                                                          GLM-Z1-9B

                                                                                         10
                                                                                                     DS-Qwen-14B          DS-Qwen-7B
                                                                                          5
  5    Anticipating the Penalty Across Models                                             0       DS-Llama-8B

Whether answer conditioning harms a model is neither uni-                                                                                             r = 0.96, n = 8
                                                                                         −5

versal nor random. It is anticipated, before any fine-tuning,                                 0           5          10       15        20       25           30
                                                                                                         Rationalization signature ΔAFR (pp)
by ∆AFR = AFRleaked − AFRblind , which measures how
much more often the model states the gold answer early when
it can see it. Computing it requires only a small unlabeled        Figure 4: The ∆AFR signature against the answer-leakage
sample of generations from the candidate teacher, with no          penalty across eight thinking models from four families. The
labels beyond the training answers already in hand.                dashed line is the linear fit and the shaded band its dispersion.
                                                                   Three points are out-of-sample, and labels are colored by in-
The answer-first signature orders the penalty across four          sample against held-out. Error bars show sampling variability
models. Table 3 shows that the larger a model’s ∆AFR,              on both axes.
the larger its penalty. The two track together across the first
four thinking models at r = 0.96 on a small sample. Figure 4
plots the same relation.
                                                                   reported in Appendix I.
   The penalties run from a substantial cost on Qwen3-8B
through an intermediate cost on DeepSeek-R1-Distill-Qwen-
7B to a negligible one on DeepSeek-R1-Distill-Llama-8B.            The answer-first signature survives the capability and
The penalty therefore scales continuously with the signature       family confounds. Appendix I reports that the signature
rather than switching on for some models and off for oth-          survives two covariates that could explain the relation away.
ers. One further thinking model, Phi-4-reasoning, cannot be        Base accuracy correlates with the penalty at 0.76, yet con-
scored for measurement reasons, so we report the fit as an         trolling for it the penalty still follows the signature at a partial
interpolation over the measured range, with the details in         correlation of 0.91, and holding out an entire family and pre-
Appendix I.                                                        dicting it from the other three gives a mean absolute error
                                                                   of 2.9 points. More fundamentally, the carrier experiment of
The fit predicts two held-out models before training them.         Section 4.3 tests the same mechanism within a single model
Table 3 marks the held-out models with bold observed val-          and corpus, where neither confound can exist.
ues. A completely different model family, NVIDIA’s Llama-
Nemotron-8B (Bercovich et al. 2025) independent of both            The leaked arm loses in both regimes, differing only in
Qwen and DeepSeek, is predicted to lose 17.4 points and            magnitude. Table 3 splits the eight models by base ac-
loses 17.6. Table 4 collects the four held-out predictions         curacy into a saturated regime, where SFT adds little and
against their observed penalties, every one landing within         leakage destroys an already-strong model, and a headroom
its predicted band, with the fits in Appendix H. Retraining        regime, where blind SFT gains but leakage forfeits part of
the two held-out predictor arms on a four-times expanded           that gain, yet blind never loses to leaked. On the headroom
matched set of 807 pairs, whose ∆AFR = +25.3 stays con-            base DeepSeek-R1-Distill-Qwen-7B blind self-distillation
sistent, gives blind 90.6 against leaked 65.8, a penalty of        reaches 89.7 over three seeds, clearing the 79.6 base, while
+24.8 within 0.4 points of the fit’s prediction. Across all        the leaked arm forfeits part of that gain at 82.4, a +7.3
eight models the signature explains the penalty closely at         penalty positive in every seed, with the seed-level scores in
r = 0.960, and no single point carries it, with the intervals      Appendix H.
Setting                         Blind ↑    Leaked ↑    Penalty ↓    so we excise from each Leaked-Early chain the single line
                                                                    where the gold answer is first stated, a median cut of 302 char-
(A) Code domain, Qwen3-8B student
                                                                    acters or about 1.4% of the chain, and retrain the otherwise-
MBPP+ (pass rate)             68.3           58.8           9.5
                                                                    identical 438 chains. The excised arm scores 88.8, recov-
HumanEval+ (pass rate)        72.8           63.6           9.2
                                                                    ering 24.6 of the 29.8 points that separate Leaked-Early
(B) Smaller student                                                 from its blind twin, or 83% of the gap, and surpasses even
Cross-student (1.7B from 8B)      85.7       78.9           6.9     the derivation-first leaked arm, so the early answer statement
                                                                    itself is the dominant carrier.
(C) Cross-family teacher, Qwen3-8B student
Nemotron                        94.0       79.2            14.8     The penalty extends past boxed-answer math. Table 5
GLM                             93.0       82.8            10.2     shows that the penalty survives in code, where the “answer” is
                                                                    the reference solution and “correct” means passing the tests.
Table 5: The one-bit intervention carried to the code domain,       On MBPP+ the penalty is about 9 to 10 points over three seeds
a smaller student, and two further teacher families with a          and HumanEval+ echoes it, so answer-conditioned chains
Qwen3-8B student. Columns are Blind and Leaked accuracy             that pass tests still harm the student. The chain-level signa-
and the red Penalty is Blind − Leaked, MATH-500 accuracy            ture transfers as well, leaked code chains echoing a verbatim
for the math rows and pass rate for the code rows.                  fragment of the reference solution early in their reasoning
                                                                    13.0% of the time against 1.3% for blind chains. A length-
                                                                    matched control rules out length for the code domain, as
                   6    Generalization                              Appendix H details.
                                                                       The penalty does not survive onto multiple-choice knowl-
Beyond the four math benchmarks of Table 1(C), the one-bit          edge and science benchmarks, where little derivation is
experiment still fixes one student that also serves as generator,   needed. On MMLU and GPQA-Diamond the blind and
one recipe, and one domain, and relaxing each in turn leaves        leaked students sit together near the base model as three-
the effect intact. Table 5 collects those relaxations, where        seed means, the differences within noise, with the values in
the penalty holds with the same sign across every generative        Appendix H. The boundary is the one the mechanism pre-
setting and vanishes only where the task needs no multi-step        dicts, since these tasks select an option rather than generate
derivation, the boundary the mechanism predicts.                    a derivation.
The damage travels with the data, not the student. Ap-
pendix H reports that the harm follows the corpus when                                    7    Conclusion
a strong teacher’s chains train a different, smaller student.       A correct final answer does not make a model-generated
Fine-tuning Qwen3-1.7B on the Qwen3-8B corpora over the             chain of thought a safe distillation target, because the correct-
same 935 problems still leaves the leaked arm about 7 points        ness filter cannot see how the chain was produced. Varying
below blind on average across three seeds, though the insta-        only whether generation is answer-conditioned in an other-
bility reflects small-student fine-tuning variance rather than      wise identical pipeline costs 16.2 MATH-500 points. The
the data. STaR-style rescue (Zelikman et al. 2022) that leaks       damage rides on rationalized chains that state the known an-
the answer only on failed problems still costs a real 4.0-point     swer early instead of deriving it, and the carrier experiment
hit, as Table 11 reports, so even verified rescued chains are       attributes most of the harm to exactly those chains. The same
negative-value data.                                                habit is legible before any training, since ∆AFR read off
                                                                    a small sample of unlabeled generations orders the penalty
Two unrelated teacher families carry the same penalty.              across eight models from four families, so a candidate teacher
Table 5 reports the penalties that chains from two further          can be screened before any fine-tuning compute is spent.
teacher families impose on an unrelated Qwen3-8B student.              Where the answer must be shown, a derive-first instruc-
When a Llama-3.1-Nemotron-Nano-8B teacher, a from-                  tion recovers about two-thirds of the penalty while keeping
scratch Llama lineage distinct from Qwen and DeepSeek,              every example. Generating answer-blind avoids the pathol-
supplies the chains, its leaked corpus costs the student a 14.8-    ogy outright, holding base accuracy within seed noise at 95.1
point penalty. A GLM-Z1-9B teacher, an unrelated fourth             against a 95.2 base while beating gold-data SFT at 90.8 under
family, costs a 10.2-point penalty. In the predictor of Table 4     a matched budget. Verifiable-reasoning distillation pipelines
each model both wrote and learned its own chains, so teacher        should therefore audit how their chains were generated, not
and student were confounded, whereas here the teacher hands         only whether the chains end at the right answer.
its corpus to a model of a separate family, which isolates the
corpus. A length-matched control on the GLM transfer rules                               A     Limitations
out length, as Appendix H details. Whether the student is the
teacher itself, a smaller sibling, or a model from a separate       Scope and coverage. The one-bit experiment uses 935
family, one corpus of answer-conditioned chains lowers the          matched math problems on one architecture family at a time,
accuracy of every student it trains.                                with the content-versus-length attribution on MATH-500 and
                                                                    the cross-model fit small-sample at eight models. The single-
A single answer-stating line carries most of the harm.              seed rows of Table 5 support a consistent sign rather than an
The carrier experiment of Section 4.3 marks the answer-first        exact magnitude, and the code domain and the student-output
population, yet correlated properties could still ride along,       transfer of Section 4 are not length-controlled.
         AIME 2024–2025       Blind    Leaked     Penalty            Subset                         Base Blind Leaked        ∆
         Seed 42               75.0     48.3         26.7            Whole set                      38.7   36.5    48.7    +12.2
         Seed 7777             65.0     38.3         26.7            All correct (387)              37.1   35.8    45.2    +9.4
         Seed 123              75.0     46.7         28.3            Both correct, 2 students (390) −      35.6    45.4    +9.8
         Mean                  71.7      44.4        27.2
                                                                   Table 8: Student-output answer-first rate (AFR, Eq. (2)) on
                                                                   each student’s own MATH-500 evaluation chains under the
Table 6: Seed-level AIME 2024–2025 accuracy of the head-
                                                                   answer-blind solve prompt, by subset, where ∆ is leaked
line blind and leaked Qwen3-8B fine-tunes on 60 off-
                                                                   minus blind. Covered chains number 189/488 for base,
distribution competition problems, with Penalty the Blind
                                                                   181/496 for blind, and 223/458 for leaked on the whole
minus Leaked difference and the bottom row the three-seed
                                                                   set.
mean.

                                             Blind     Leaked      uation chains under the standard answer-blind solve prompt
     (a) Position and length on the 935 matched chains             against each chain’s own boxed answer. Coverage below 500
     Median first-answer position           0.587    0.297         comes from chains that lack a parseable boxed answer or a
     Answer within first 5%                  13.3     41.9
                                                                   complete think block. The leaked student states its answer
                                                                   earlier even on the all-correct subset that matches difficulty
     Median think-block length (K chars)     24.7     18.2
                                                                   across the three students.
     Median first mention (K chars)          12.2      4.5
                                                                      A length caveat applies, since leaked chains average
     Answer coverage (of 935)               ≥ 918 ≥ 918
                                                                   17,882 characters against 13,884 for blind. Under an
     (b) Answer-first rate by absolute character prefix            absolute-character window the matched-difficulty gap nar-
     Within 1K chars                          12.2      42.0       rows to between 2.3 and 4.4 points, though the median first-
     Within 2K chars                          17.1      44.8       answer position stays earlier for the leaked student at 0.238
     Within 4K chars                          25.8      48.9       against 0.287.
     Within 8K chars                          38.3      59.0          Across its 62 incorrect chains the leaked student is answer-
                                                                   first 74.2% of the time against 44.7% on its correct chains,
Table 7: Answer-position statistics on the 935 matched chain       so the both-correct control is the trustworthy comparison.
pairs of the mechanism analysis. Positions in (a) are relative     Across the three training seeds {42, 123, 7777} the whole-
fractions inside each chain’s own reasoning, and (b) reports       set gap is 12.2, 15.8, and 11.4 points and the two-student
answer-first rates at absolute character prefixes.                 both-correct gap is 9.8, 14.1, and 8.1, positive in every seed,
                                                                   so the transfer is robust in sign.

B     Seed-Level Difficulty and AIME Numbers                                  E     Evaluation-Budget Sweep
Table 6 reports the seed-level AIME numbers behind the             The budget sweep of Figure 3 re-tokenizes every evaluation
three-seed mean in Table 1(C). The pooled hardest MATH-            response before applying each budget cutoff. The penalty
500 band (Level 4–5, 262 problems) gives blind 92.6 against        rises to its peak at the 8K budget and never closes as the bud-
leaked 72.5, a 20.1-point penalty, where the blind arm has         get extends to the 32K cap, which rules out a budget-artifact
real headroom rather than sitting at ceiling, with the seed        reading. Across the three training seeds {42, 123, 7777} the
dispersions in Appendix I.                                         penalty at the 32K cap is 15.6, 16.6, and 16.4 points, the
                                                                   derivation deficit is 9.5, 9.1, and 7.3, and the sweep peaks
           C     Answer-Position Statistics                        at the 8K budget in every seed, so the shape is robust across
Table 7 reports answer-position statistics for the matched         seeds. The Qwen3-4B curve is uniformly higher, mirroring
pairs of Section 4, as relative positions inside each chain, the   its larger overall penalty. The significance tests for these gaps
faithful unit for a rationalization claim, and as answer-first     are in Appendix I.
rates at absolute character cutoffs. The leaked chains state the
gold answer earlier on every measure, so the leaked-against-            F     Chain-Population Split Robustness
blind ordering is the same under a relative or an absolute         Table 9 shows the carrier split of Section 4.3 replicat-
window. The leaked chain states its answer earlier than its        ing at two further training seeds, the Leaked-Late-minus-
blind twin on 75 of every hundred problems by absolute po-         Leaked-Early gap staying decisively positive at all three.
sition, and 57 of every hundred think blocks are shorter than      The Leaked-Early model produces no extractable answer
the 20%-of-budget window in full, so an absolute-budget            on 15.4% of evaluation problems against 1.6% for Leaked-
threshold cannot localize an early statement.                      Late and 1.2% and 0.4% for the blind arms, the termination
                                                                   pathology concentrating where the mechanism predicts.
     D     Student-Output Answer-First Rate                          The 438/497 partition is not knife-edge. A strict an-
Table 8 expands the student-output transfer of Section 4,          swer matcher agrees with the default on 98.9% of the
scoring AFR (Eq. (2)) on each student’s MATH-500 eval-             935 leaked chains. Sweeping the answer-first threshold over
        Seed   Leaked-Early      Leaked-Late      Gap              Benchmark     Arm      Seed 42    Seed 123   Seed 7777    Mean
        42           64.2             86.2       +22.0             MATH-500      Blind      95.2       94.6        95.4       95.1
        123          71.8             85.0       +13.2                           Leaked     79.6       78.0        79.0       78.9
        7777         66.0             84.0       +18.0             GSM8K         Blind      95.6       95.2        95.4       95.4
                                                                                 Leaked     89.6       90.4        91.6       90.5
Table 9: Seed-level carrier arms, MATH-500 accuracy of the         Minerva       Blind      48.2       48.5        49.3       48.7
                                                                                 Leaked     36.8       40.8        37.9       38.5
Leaked-Early and Leaked-Late fine-tunes at each train-
ing seed, with Gap the Leaked-Late minus Leaked-Early
difference.                                                        Table 10: Seed-level accuracies of the headline blind and
                                                                   leaked Qwen3-8B fine-tunes on the three math benchmarks
                                                                   whose three-seed means appear in Table 1(C).
{0.1, 0.2, 0.3} shifts the early share only across 43.4%,
46.8%, and 49.6%. Eleven chains never state the answer
inside the think block and are conservatively assigned to          DeepSeek-R1-Distill-Qwen-7B 79.6, DeepSeek-R1-Distill-
Leaked-Late.                                                       Llama-8B 77.8, and DeepSeek-R1-Distill-Qwen-14B 88.2
   The blind twins of Searly problems are themselves answer-       (Qwen Team 2025; DeepSeek-AI et al. 2025).
first 48.9% of the time against 6.0% for Slate , so the split      Gold-data baseline. A raw gold-data SFT baseline trained
partially reflects problem-level properties and motivates the      on the full multi-domain corpus scores 90.8 on MATH-500
blind control arms.                                                at the matched 10,240-token budget, the gold-data reference
   The decision rules are as follows.                              that answer-blind self-distillation exceeds at the same budget.
1. The carrier hypothesis holds when the difference-in-
   differences interval excludes 0 and the two blind arms          Seed-level scores. Table 10 reports the seed-level scores
   differ by at most 3 points.                                     behind the three-seed means of Table 1(C). The MATH-
                                                                   500 blind-minus-leaked gaps of 15.6, 16.6, and 16.4 average
2. The full-carrier outcome additionally requires Leaked-          16.2, and the GSM8K and Minerva penalties average 4.9 and
   Late within 3 points of the blind arms.                         10.2.
3. The partial-carrier outcome requires Leaked-Late at
   least 5 points above Leaked-Early but more than 3               Length-matched control. For the length-matched control,
   points below blind.                                             leaked training chains run about 5,800 characters shorter than
                                                                   their blind twin, with 732 of 935 shorter. The length-matched
                                                                   blind arm scores 95.0, within 0.2 to 0.4 points of the full blind
G    Generation, Training, and Harness Details                     condition and 15.4 points above the leaked arm.
Generation. For the one-bit experiment (Section 3) we
sample n=2 chains for each problem at temperature 0.6,              H     Additional Details for the Generalization
top-p 0.95, top-k 20, with a 16,384-token thinking budget,                             Experiments
from Qwen3-8B. The Leaked prompt appends the gold an-
swer and requests the reasoning that reaches it, and the Blind     This section expands the setups and secondary statistics for
prompt is the standard solve instruction. We keep the first cor-   the relaxations of Section 6, and all headline numbers appear
rect chain for each problem, with the final boxed answer ver-      in the main text.
ified by symbolic matching, intersect the problem sets solved
under both conditions to obtain the 935 matched problems,          Third-family point. For the independent third family of
and fine-tune each condition on (question → chain) pairs           Section 5 we use NVIDIA’s Llama-3.1-Nemotron-Nano-
with identical training inputs. These problems are drawn           8B, a from-scratch Llama lineage distinct from Qwen and
from the math split of our multi-domain SFT training corpus        DeepSeek. Generation needs the system instruction detailed
and are disjoint from the held-out MATH-500 test set on            thinking on to elicit a think block, after which the stan-
which all models are evaluated, so the comparison reflects         dard solve harness applies unchanged, and the fine-tuned
generalization to unseen problems rather than memorization         arms emit a think block and a boxed answer on their own
of evaluated items.                                                at evaluation. On 855 matched problems the answer-first
                                                                   rates are 22.1% blind and 40.0% leaked, a signature of
Training and evaluation. All SFT runs use DeepSpeed                ∆AFR = +17.9. Full SFT of each arm under the same
ZeRO-3 (Rajbhandari et al. 2020), 1 epoch, sequence length         recipe and harness as the other points gives blind 92.2 against
10,240, learning rate 2 × 10−5 , and effective batch size 48,      leaked 74.6, a penalty of +17.6, within 0.2 points of the
and all Qwen3-8B-arm models are full-parameter SFT of              +17.4 predicted from the six-model fit before training. The
Qwen3-8B. Evaluation uses vLLM thinking mode at tem-               other held-out predictions were fixed from a four-model fit
perature 0.6, top-p 0.95, top-k 20, and MATH-500 uses a            for Qwen3-1.7B and DS-R1-Distill-Qwen-14B and from a
32,768-token generation budget and symbolic answer match-          seven-model fit for the fourth-family GLM-Z1-9B. Table 4
ing. The base-model MATH-500 accuracies under this har-            collects the predicted and observed pairs, every one inside
ness are Qwen3-8B 95.2, Qwen3-4B 96.0, Qwen3-1.7B 92.6,            its band of about ±9 points.
Base          Answer-cond.    Blind-only ↑    STaR-rescue ↑       Cost ↓   Setting                    Blind ↑   Length-matched ↑   Leaked ↑
Qwen3-8B          45.8%          95.2               91.2           4.0     GLM teacher (MATH-500)      93.0           90.9           82.8
DS-Qwen-7B        27.6%          92.2               90.7           1.5     Code (MBPP+ pass rate)      70.8           70.7           59.2

Table 11: STaR-replica rescue cost on MATH-500. Each cor-                  Table 13: Length-matched controls for the cross-family and
pus adds answer-hint rescues of the blind failures to the                  code transfers, MATH-500 accuracy for the GLM row and
blind correct chains, so a fraction of it is answer-conditioned.           MBPP+ pass rate for the code row. Each control truncates
Blind-only trains on the blind chains alone, STaR-rescue on                every blind chain to its matched leaked chain’s length and
the combined corpus, both filtered for correctness. Cost is                retrains the student under the same recipe. Every code cell
Blind-only minus STaR-rescue. The DS-Qwen-7B row is a                      and the length-matched cells are means over the two seeds
two-seed mean.                                                             {42, 7777} of the control, and the GLM Blind and Leaked
                                                                           cells reproduce the corresponding arms of Table 5.
       Reference-echo rate (%)           Blind        Leaked
       Early (first 20% of reasoning)         1.3          13.0
       Ever (anywhere in the chain)          12.2          92.1            if anything shorter, at a mean 5,954 characters against the
                                                                           leaked 8,378. Its seed scores are 70.9 and 70.4 over seeds
                                                                           {42, 7777}, far above the matched leaked arm, which shows
Table 12: Reference-echo rates of the blind and leaked code
                                                                           content rather than length drives the code penalty.
chains on the 379 matched MBPP training pairs. A chain
echoes when its reasoning contains a verbatim fragment of
the reference solution, counted within the first 20% of the                Multiple-choice benchmarks. The same headline blind
reasoning (early) or anywhere in the chain (ever). The tinted              and leaked Qwen3-8B fine-tunes evaluated on multiple-
column is the answer-blind arm.                                            choice benchmarks show no penalty across three seeds, con-
                                                                           sistent with the harm riding on multi-step derivation. On
                                                                           GPQA-Diamond, over its 198 problems, the three-seed blind
Cross-student transfer. We fine-tune Qwen3-1.7B across                     and leaked means are 37.6 and 40.2 against a base of 41.9,
three seeds on the Qwen3-8B-generated Blind and Leaked                     and on a 2,000-problem MMLU subset they are 54.8 and
corpora built from the same 935 problems, filter, and recipe.              54.0 against a base of 54.9. Both arms sit near base and nei-
Table 5 reports the resulting +6.9-point mean penalty, whose               ther difference is distinguishable from zero, so on tasks that
seed penalties are 16.8, −0.2, and 4.0, with the dispersion                need little derivation the answer-leakage penalty is absent,
reported in Appendix I. The small-student transfer therefore               and the seed-level values appear in Appendix I.
yields a positive penalty on average but is the least stable set-
ting we test, in contrast to the seed-robust cross-family trans-           Cross-family length control. For the GLM-Z1-9B teacher
fers. We read the average penalty as tracking the teacher’s                we repeat the length-matched control of Section 3, truncating
∆AFR of +20.8 rather than the student’s +26.2, consistent                  each blind chain to its matched leaked length and retraining
with leakage being a property of the data.                                 a Qwen3-8B student under the same recipe. Table 13 reports
                                                                           the arm, whose seed scores of 90.8 and 91.0 over seeds
STaR-replica regime. We combine blind correct chains                       {42, 7777} sit near the untruncated blind arm and far above
with answer-hint rescues of the failures, forming a corpus that            the leaked one, so the cross-family gap is content, not length.
is partly answer-conditioned, and Table 11 reports the cost.
A linear dose account would predict a cost proportional to
                                                                           Headroom regime. On DeepSeek-R1-Distill-Qwen-7B,
the answer-conditioned fraction, and the measured 4.0-point
                                                                           whose base is 79.6, the STaR-replica arm adds answer-hint
cost is real but somewhat sub-linear in that dose, plausibly
                                                                           rescues of the blind failures, and Table 11 reports the re-
because the rescued problems also contribute genuinely new
                                                                           sulting answer-conditioned fraction and its 1.5-point mean
content, which is the STaR rationale for rescuing them.
                                                                           cost over two seeds, positive in both. The one-bit experiment
Code domain. From the 474 MBPP training problems,                          on the same base, blind against leaked self-distilled chains
with the test split unused, Qwen3-8B generates solutions                   over 348 matched problems, scores blind 91.4/89.2/88.4
blind and with the reference shown. Both arms keep only                    for a mean of 89.7 against leaked 86.4/80.2/80.6 for a
chains whose extracted code passes all asserts, and we train               mean of 82.4 over seeds {42, 123, 7777}. The penalties of
on the 379 matched pairs. Table 12 reports the chain-level                 5.0/9.0/7.8, a mean of 7.3, stay positive in every seed even
signature, whose early-echo gap of ∆echo = +11.7 points                    though blind training gains about ten points over the base.
is the code-domain analogue of the early-answer behavior
in math. The MBPP+ penalty, over its 378 problems, and                     Derive-first mitigation. Regenerating the leaked chains
the HumanEval+ transfer penalty, over its 164 problems, are                with the gold answer in context but an explicit instruction
three-seed means reported in Table 5 and positive in every                 to derive from scratch and reveal the answer only at the end
seed, with the MBPP+ seed values in Appendix I. Table 13                   produces the derive-first arm of Table 1(A). Its answer-first
reports a length-matched control that truncates each blind                 rate sits between the leaked and blind arms, the shift Section 4
chain to its matched leaked length, so the control chains run              reports.
Quantity                                    Value            Note                                held-out points share our generation and evaluation harness,
Signature–penalty correlation r             0.960            exact permutation p = 0.0003        so the fit should be read as a strong but small-sample trend
Leave-one-out range of r                    [0.949, 0.981]   dropping one model at a time        rather than a precisely estimated law. The leave-one-family-
Fisher-z 95% interval on r                  [0.79, 0.99]     wide at eight models
Partial corr., controlling base acc.        0.91             permutation p = 0.0019              out row shows the relation is not merely a Qwen-versus-
  bootstrap 95% interval                    [0.79, 1.00]     five effective degrees of freedom   DeepSeek family label, and the exclusion row shows the fit
Partial corr., controlling blind-arm acc.   0.96                                                 does not hang on its lowest-confidence anchor.
Leave-one-family-out error                  2.9 points       single-model families within 1.7
                                            r = 0.955        base-acc. partial rises to 0.93
Excluding DS-R1-Distill-Llama-8B
                                                                                                 Code domain. The MBPP+ code-domain penalty is a
                                                                                                 three-seed mean, with blind 68.3 (standard deviation 6.3)
Table 14: Inferential statistics for the cross-model signature                                   against leaked 58.8 (standard deviation 4.4) and a mean
fit, the relation between the pre-fine-tuning ∆AFR signature                                     penalty of 9.5 points over seeds {42, 123, 7777}, with the
of Eq. (2) and the leakage penalty across the eight thinking                                     three seed penalties 11.9, 5.3, and 11.3. The HumanEval+
models of Table 3. The leave-one-family-out row fits on three                                    transfer is reported at 1.6 standard errors.
families and predicts the entire held-out family, reporting
mean absolute error in penalty points.                                                           Cross-student transfer. The cross-student transfer penalty
                                                                                                 has a seed standard deviation of 8.9.

        I      Statistical Significance, Confidence                                              Multiple-choice benchmarks. On the two multiple-choice
                                                                                                 benchmarks the three-seed blind-minus-leaked means are
                        Intervals, and Tests                                                     −2.6 points on GPQA-Diamond and +0.8 on MMLU, so
This section is the sole home for all confidence intervals,                                      the leaked arm is marginally ahead on GPQA-Diamond and
bootstrap and difference-in-differences intervals, standard                                      marginally behind on MMLU, and neither sign is distinguish-
deviations, standard errors, and significance tests reported                                     able from zero across seeds {42, 123, 7777}, so we report no
anywhere in the paper. Each block below covers one ex-                                           degradation rather than a reversed effect.
periment, and the point estimates and descriptive ranges it
annotates appear in the matching body section.                                                   Off-distribution AIME. The off-distribution AIME
                                                                                                 penalty is a three-seed mean of 27.2 points, with blind
Shared tests. Accuracy gaps and the difference-in-                                               and leaked standard deviations of 5.8 and 5.4 over seeds
differences endpoint use a paired bootstrap over the eval-                                       {42, 7777, 123}.
uation problems, and discordant-item asymmetries use Mc-
Nemar’s test.                                                                                    Phi-4-reasoning. Phi-4-reasoning is the single thinking
                                                                                                 model tried and set aside for measurement reasons (Sec-
One-bit experiment. For the one-bit experiment the dis-                                          tion 5). It carries the largest signature in the set at ∆AFR =
cordant items split 86 to 8 in favor of the blind arm (p <                                       +45.5, and the frozen six-model fit predicts a 47.4-point
10−17 ), the paired-bootstrap 95% interval on the penalty is                                     penalty. Both trained arms, however, collapse to under 5%
[12.2, 19.2] points, and the three-seed standard deviations                                      MATH-500 accuracy, boxing an answer for only 3 and 31
are 0.4 for the blind arm and 0.8 for the leaked arm. On                                         of 500 problems while rambling past the 32K budget in de-
the pooled MATH-500 Level 4–5 band the blind and leaked                                          generate loops with a mean response over 50K characters.
seed dispersions are ±0.2 and ±0.8 with non-overlapping                                          The realized gap is therefore degenerate rather than a valid
seed ranges.                                                                                     penalty, and the point is discarded rather than graded favor-
Answer position. The answer-position shift is significant                                        ably. We report the fit as an interpolation over the measured
by a two-sample Kolmogorov–Smirnov test (D = 0.291,                                              ∆AFR range rather than an extrapolation that this single
p < 10−34 ).                                                                                     unanchored extreme could confirm or distort.

Budget sweep. In the budget sweep the both-completed                                                               J   Reproducibility
derivation gap and the Qwen3-4B completed-only gap are
                                                                                                 Release contents. We release the code and data needed to
significant by McNemar’s test (51 to 7, p = 2.4 × 10−9 , and
                                                                                                 regenerate every reported number. The release provides the
p = 1.9 × 10−26 ).
                                                                                                 exact generation prompts for all five conditions, the matched
Carrier experiment. In the carrier experiment, the two                                           935 problem identifiers together with their answer-blind and
leaked arms differ decisively (129 to 19, p = 2.8 × 10−21 )                                      answer-leaked chains, the generation metadata for each chain,
while the two blind controls do not (p = 0.48), the difference-                                  the full training configuration, and the evaluation harness.
in-differences carries bootstrap p < 10−4 with a single-seed
95% interval of [16.4, 25.8], and against their own blind                                        Correctness-critical components. The symbolic answer
twins the answer-first and derivation-first arms lose 29.8 (p =                                  checker that decides which generated chains pass the keep-
1.6 × 10−38 ) and 8.6 (p = 6.0 × 10−8 ) points. For Qwen3-                                       correct filter and which evaluation answers score as correct
4B the leaked arm splits the discordant items 149 to 3 (p =                                      is the one correctness-critical component. Its dependencies
2 × 10−40 ).                                                                                     are pinned to exact versions rather than version floors, so
                                                                                                 the keep-correct and evaluation decisions reproduce exactly.
Cross-model signature. Table 14 collects the inferential                                         The release also exposes the answer-position matcher whose
statistics for the cross-model signature fit of Section 5. With                                  threshold-0.2 answer-first rate defines the chain-level signa-
only eight models the Fisher-z interval on r is wide, and the                                    ture.
Traceability. A manifest maps every benchmark, condi-               Kazemi, M.; Kim, N.; Bhatia, D.; Xu, X.; and Ramachandran,
tion, and seed cell to the stored evaluation record that pro-       D. 2023. LAMBADA: Backward Chaining for Automated
duced it, so each table entry is traceable to its source. The one   Reasoning in Natural Language. In Proceedings of the 61st
setting a reader must pass explicitly is the training sequence      Annual Meeting of the Association for Computational Lin-
length 10,240, which every run uses. The held-out penalty           guistics (Volume 1: Long Papers), 6547–6568.
predictions of Section 5 are the values we recorded before          Kwon, W.; Li, Z.; Zhuang, S.; Sheng, Y.; Zheng, L.; Yu, C. H.;
training each held-out model. The released code recomputes          Gonzalez, J. E.; Zhang, H.; and Stoica, I. 2023. Efficient
the cross-model fit from the stored evaluation files, so it re-     Memory Management for Large Language Model Serving
produces the signature relationship and the leave-one-family-       with PagedAttention. In Proceedings of the ACM SIGOPS
out error rather than re-emitting the frozen point predictions.     29th Symposium on Operating Systems Principles, 611–626.
Neutral-arm comparability. The two neutral prompt-                  Lanham, T.; Chen, A.; Radhakrishnan, A.; et al. 2023. Mea-
ablation arms in Table 1 are trained with a numerically             suring Faithfulness in Chain-of-Thought Reasoning. arXiv
equivalent memory optimization and are reported as three-           preprint arXiv:2307.13702.
seed means. A matched answer-blind arm trained under that
                                                                    Li, C.; Xue, M.; Zhang, Z.; Yang, J.; Zhang, B.; Yu, B.; Hui,
optimization scores 96.2 at seed 42, within about one point
                                                                    B.; Lin, J.; Wang, X.; and Liu, D. 2025. START: Self-taught
of the standard answer-blind reference, so the optimization
                                                                    Reasoner with Tools. In Proceedings of the 2025 Conference
does not materially shift accuracy and the two neutral arms
                                                                    on Empirical Methods in Natural Language Processing.
are comparable to the standard answer-blind condition.
                                                                    Li, C.; Yuan, Z.; Yuan, H.; Dong, G.; Lu, K.; Wu, J.; Tan,
                        References                                  C.; Wang, X.; and Zhou, C. 2024. MuggleMath: Assessing
                                                                    the Impact of Query and Response Augmentation on Math
Agarwal, R.; Vieillard, N.; Zhou, Y.; Stańczyk, P.; Ramos, S.;
                                                                    Reasoning. In Proceedings of the 62nd Annual Meeting of
Geist, M.; and Bachem, O. 2024. On-Policy Distillation of
                                                                    the Association for Computational Linguistics (Volume 1:
Language Models: Learning from Self-Generated Mistakes.
                                                                    Long Papers).
In International Conference on Learning Representations
(ICLR).                                                             Li, G.; and Zhang, X. 2026. K-STaR: Knowledge-Aware Self-
Bercovich, A.; Levy, I.; Golan, I.; Dabbah, M.; El-Yaniv, R.;       Taught Reasoner. In Proceedings of the AAAI Conference on
Puny, O.; et al. 2025. Llama-Nemotron: Efficient Reasoning          Artificial Intelligence.
Models. arXiv preprint arXiv:2505.00949.                            Li, Y.; Shi, X.; Kou, Q.; Yu, L.; and Zhou, H. 2026.
Chen, J. C.-Y.; Wang, Z.; Palangi, H.; et al. 2024. Reverse         RAFT: Data Refinement and Adaptive Distillation for Do-
Thinking Makes LLMs Stronger Reasoners. arXiv preprint              main Fine-Tuning with Alleviated Forgetting. arXiv preprint
arXiv:2411.19865.                                                   arXiv:2606.00147.
Chen, Y.; Benton, J.; Radhakrishnan, A.; Uesato, J.; Denison,       Lightman, H.; Kosaraju, V.; Burda, Y.; Edwards, H.; Baker,
C.; Schulman, J.; Somani, A.; Hase, P.; Wagner, M.; Roger,          B.; Lee, T.; Leike, J.; Schulman, J.; Sutskever, I.; and
F.; Mikulik, V.; Bowman, S. R.; Leike, J.; Kaplan, J.; and          Cobbe, K. 2023. Let’s Verify Step by Step. arXiv preprint
Perez, E. 2025. Reasoning Models Don’t Always Say What              arXiv:2305.20050.
They Think. arXiv preprint arXiv:2505.05410.                        Luo, L.; Liu, Y.; Liu, R.; Phatale, S.; Guo, M.; Lara, H.;
Cobbe, K.; Kosaraju, V.; Bavarian, M.; Chen, M.; Jun, H.;           Li, Y.; Shu, L.; Zhu, Y.; Meng, L.; Sun, J.; and Rastogi,
Kaiser, L.; Plappert, M.; Tworek, J.; Hilton, J.; Nakano, R.;       A. 2024. Improve Mathematical Reasoning in Language
Hesse, C.; and Schulman, J. 2021. Training Verifiers to Solve       Models by Automated Process Supervision. arXiv preprint
Math Word Problems. arXiv preprint arXiv:2110.14168.                arXiv:2406.06592.
DeepSeek-AI; Guo, D.; Yang, D.; Zhang, H.; Song, J.;                Luong, T. Q.; Zhang, X.; Jie, Z.; Sun, P.; Jin, X.; and Li,
Wang, P.; Zhu, Q.; Xu, R.; Zhang, R.; Ma, S.; Bi, X.;               H. 2024. ReFT: Reasoning with Reinforced Fine-Tuning. In
et al. 2025. DeepSeek-R1: Incentivizing Reasoning Capa-             Proceedings of the 62nd Annual Meeting of the Association
bility in LLMs via Reinforcement Learning. arXiv preprint           for Computational Linguistics (Volume 1: Long Papers).
arXiv:2501.12948.
                                                                    OpenAI. 2024. OpenAI o1 System Card.
Gulcehre, C.; Paine, T. L.; Srinivasan, S.; et al. 2023. Rein-
forced Self-Training (ReST) for Language Modeling. arXiv            Paul, D.; West, R.; Bosselut, A.; and Faltings, B. 2024. Mak-
preprint arXiv:2308.08998.                                          ing Reasoning Matter: Measuring and Improving Faithful-
                                                                    ness of Chain-of-Thought Reasoning. In Findings of the
Hendrycks, D.; Burns, C.; Kadavath, S.; Arora, A.; Basart,
                                                                    Association for Computational Linguistics: EMNLP 2024.
S.; Tang, E.; Song, D.; and Steinhardt, J. 2021. Measuring
Mathematical Problem Solving With the MATH Dataset. In              Phan, D.; Hoffman, M. D.; Dohan, D.; Douglas, S.; Le,
Thirty-fifth Conference on Neural Information Processing            T. A.; Parisi, A.; Sountsov, P.; Sutton, C.; Vikram, S.; and
Systems Datasets and Benchmarks Track.                              A Saurous, R. 2023. Training chain-of-thought via latent-
Hosseini, A.; Yuan, X.; Malkin, N.; Courville, A.; Sordoni,         variable inference. Advances in Neural Information Process-
A.; and Agarwal, R. 2024. V-STaR: Training Verifiers for            ing Systems, 36: 72819–72841.
Self-Taught Reasoners. In First Conference on Language              Qwen Team. 2025. Qwen3 Technical Report. arXiv preprint
Modeling (COLM). ArXiv:2402.06457.                                  arXiv:2505.09388.
Rajbhandari, S.; Rasley, J.; Ruwase, O.; and He, Y. 2020.      Correlations Between Question and Answer via Chain-of-
ZeRO: Memory Optimizations Toward Training Trillion Pa-        Thought Correctness Perception Distillation. arXiv preprint
rameter Models. In SC20: International Conference for High     arXiv:2509.05602.
Performance Computing, Networking, Storage and Analysis,       Xiong, F.; Xu, H.; Wang, Y.; Cheng, R.; Wang, Y.; and Chu,
1–16. IEEE.                                                    X. 2025. HS-STaR: Hierarchical Sampling for Self-Taught
Sang, H.; Xu, Y.; Zhou, Z.; He, R.; Wang, Z.; and Sun, J.      Reasoners via Difficulty Estimation and Budget Realloca-
2026. Crisp: Compressed reasoning via iterative self-policy    tion. In Proceedings of the 2025 Conference on Empirical
distillation. arXiv preprint arXiv:2603.05433.                 Methods in Natural Language Processing.
Singh, A.; Co-Reyes, J. D.; Agarwal, R.; et al. 2023. Beyond   Xu, R.; Qi, Z.; and Xu, W. 2024. Preemptive Answer “At-
Human Data: Scaling Self-Training for Problem-Solving          tacks” on Chain-of-Thought Reasoning. In Findings of the
with Language Models. arXiv preprint arXiv:2312.06585.         Association for Computational Linguistics: ACL 2024.
Song, M.; Su, Z.; Qu, X.; Zhou, J.; and Cheng, Y. 2025.        Yu, L.; Jiang, W.; Shi, H.; Yu, J.; Liu, Z.; Zhang, Y.; Kwok,
PRMBench: A Fine-grained and Challenging Benchmark for         J. T.; Li, Z.; Weller, A.; and Liu, W. 2024. MetaMath: Boot-
Process-Level Reward Models.                                   strap Your Own Mathematical Questions for Large Language
Turpin, M.; Michael, J.; Perez, E.; and Bowman, S. R. 2023.    Models. In International Conference on Learning Represen-
Language Models Don’t Always Say What They Think: Un-          tations (ICLR). ArXiv:2309.12284.
faithful Explanations in Chain-of-Thought Prompting. In        Yuan, Z.; Yuan, H.; Li, C.; Dong, G.; Lu, K.; Tan, C.; Zhou,
NeurIPS.                                                       C.; and Zhou, J. 2023. Scaling Relationship on Learn-
                                                               ing Mathematical Reasoning with Large Language Models.
Tutek, M.; Hashemi Chaleshtori, F.; Marasović, A.; and Be-
                                                               arXiv preprint arXiv:2308.01825.
linkov, Y. 2025. Measuring Chain of Thought Faithfulness
by Unlearning Reasoning Steps. In Proceedings of the 2025      Zelikman, E.; Harik, G.; Shao, Y.; Jayasiri, V.; Haber, N.; and
Conference on Empirical Methods in Natural Language Pro-       Goodman, N. D. 2024. Quiet-STaR: Language Models Can
cessing.                                                       Teach Themselves to Think Before Speaking. arXiv preprint
                                                               arXiv:2403.09629.
Uesato, J.; Kushman, N.; Kumar, R.; Song, F.; Siegel, N.;
Wang, L.; Creswell, A.; Irving, G.; and Higgins, I. 2022.      Zelikman, E.; Wu, Y.; Mu, J.; and Goodman, N. D. 2022.
Solving math word problems with process- and outcome-          STaR: Bootstrapping Reasoning With Reasoning. In Ad-
based feedback. arXiv preprint arXiv:2211.14275.               vances in Neural Information Processing Systems, vol-
                                                               ume 35, 15476–15488. Curran Associates, Inc.
Wang, B.; Lee, C.; Lee, N.; Lin, S.-C.; Dai, W.; Chen, Y.;
Yang, Z.; Liu, Z.; Shoeybi, M.; Catanzaro, B.; and Ping, W.    Zhang, D.; Zhoubian, S.; Hu, Z.; Yue, Y.; Dong, Y.; and
2025. Nemotron-Cascade: Scaling Cascaded Reinforcement         Tang, J. 2024. ReST-MCTS*: LLM Self-Training via Process
Learning for General-Purpose Reasoning Models. arXiv           Reward Guided Tree Search.
preprint arXiv:2512.13607.                                     Zhang, R.; Wang, L.; Zhang, H.; and Zheng, Z. 2026a. Robust
Wang, B.; Liu, Y.; Liu, Y.; Tang, T.; Wang, S.; Gao, C.;       Reasoning via Dynamic Token Selection for Distribution-
Zheng, C.; Zhang, Y.; Yu, L.; Liu, S.; Gui, T.; Zhang, Q.;     Aligned Self-Distillation. arXiv preprint arXiv:2606.00628.
Huang, X.; Yu, B.; Huang, F.; and Lin, J. 2026. Outcome        Zhang, T.; Liu, F.; Wong, J.; Abbeel, P.; and Gonzalez, J. E.
Accuracy is Not Enough: Aligning the Reasoning Process of      2023. The Wisdom of Hindsight Makes Language Models
Reward Models. In Proceedings of the 64th Annual Meeting       Better Instruction Followers. In International Conference on
of the Association for Computational Linguistics (Volume 1:    Machine Learning.
Long Papers).                                                  Zhang, W.; Yan, J.; Huang, J.; Shen, Y.; Shi, S.; Chen, P.;
Wang, P.; Li, L.; Shao, Z.; Xu, R.; Dai, D.; Li, Y.; Chen,     Wang, N.; Liu, Z.; Wang, K.; and Lian, S. 2026b. HEAL:
D.; Wu, Y.; and Sui, Z. 2024. Math-Shepherd: Verify and        Hindsight Entropy-Assisted Learning for Reasoning Distil-
Reinforce LLMs Step-by-step without Human Annotations.         lation. arXiv preprint arXiv:2603.10359.
In Proceedings of the 62nd Annual Meeting of the Associa-      Zhang, Z.; Zheng, C.; Wu, Y.; Zhang, B.; Lin, R.; Yu, B.;
tion for Computational Linguistics (Volume 1: Long Papers),    Liu, D.; Zhou, J.; and Lin, J. 2025. The Lessons of Develop-
9426–9439. Association for Computational Linguistics.          ing Process Reward Models in Mathematical Reasoning. In
Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Ichter, B.;      Findings of the Association for Computational Linguistics:
Xia, F.; Chi, E.; Le, Q. V.; and Zhou, D. 2022. Chain-         ACL 2025.
of-Thought Prompting Elicits Reasoning in Large Language       Zheng, C.; Zhang, Z.; Zhang, B.; Lin, R.; Lu, K.; Yu, B.; Liu,
Models. In Advances in Neural Information Processing Sys-      D.; Zhou, J.; and Lin, J. 2025. ProcessBench: Identifying
tems, volume 35, 24824–24837. Curran Associates, Inc.          Process Errors in Mathematical Reasoning.
Weng, Y.; Zhu, M.; Xia, F.; Li, B.; He, S.; Liu, S.; Sun,      Zhou, C.; Liu, P.; Xu, P.; et al. 2023. LIMA: Less Is More
B.; Liu, K.; and Zhao, J. 2023. Large Language Models          for Alignment. In NeurIPS.
are Better Reasoners with Self-Verification. In Findings of    Zhu, M.; Liu, Y.; Fu, Z.; Wang, Q.; and Zhang, Y. 2026.
EMNLP.                                                         In-Token Rationality Optimization: Towards Accurate and
Xie, H.; Yao, Y.; Ban, Y.; Huang, Z.; Wang, D.; Wu, Z.;        Concise LLM Reasoning via Self-Feedback. In Proceedings
Su, H.; Wang, C.; and Song, S. 2025. Mitigating Spurious       of the AAAI Conference on Artificial Intelligence.
