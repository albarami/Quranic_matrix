# QBM Annotation Decision Flowchart

## Master Decision Tree

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           START: Read the Ayah                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STEP 1: IDENTIFY THE AGENT                               │
│                                                                                 │
│  Ask: "Who is performing or experiencing this behavior?"                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
   │ Explicit     │           │ Context      │           │ General      │
   │ Marker?      │           │ Implies?     │           │ Human?       │
   └──────────────┘           └──────────────┘           └──────────────┘
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ الذين آمنوا     │         │ Previous ayat    │         │ الإنسان، الناس  │
│ → AGT_BELIEVER  │         │ mention group?   │         │ خلقكم، بني آدم  │
│                 │         │ Use that agent   │         │ → AGT_HUMAN_    │
│ الذين كفروا    │         │                  │         │   GENERAL       │
│ → AGT_DISBELIEV │         │                  │         │                 │
│                 │         │                  │         │                 │
│ المنافقين      │         │                  │         │                 │
│ → AGT_HYPOCRITE │         │                  │         │                 │
└─────────────────┘         └─────────────────┘         └─────────────────┘

                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       STEP 2: DETERMINE BEHAVIOR FORM                           │
│                                                                                 │
│  Ask: "What type of action/state is described?"                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
     ┌────────────┬────────────┬────────────┬────────────┬────────────┐
     ▼            ▼            ▼            ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Verb of │ │ Verb of │ │ قلب،   │ │ Trait   │ │ Inter-  │ │Multiple │
│ speech? │ │ action? │ │ يؤمن،  │ │ word?   │ │ personal│ │ types?  │
│ قال،   │ │ يمشي،  │ │ يخاف،  │ │ صابر،  │ │ act?    │ │         │
│ يقول   │ │ يصلي   │ │ يحب    │ │ متقي   │ │ إحسان  │ │         │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     │           │           │           │           │           │
     ▼           ▼           ▼           ▼           ▼           ▼
 speech_act  physical_  inner_state  trait_     relational   mixed
              act                   disposition    _act

                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STEP 3: IDENTIFY SPEECH MODE                             │
│                                                                                 │
│  Ask: "How is this statement structured grammatically?"                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
   │ Imperative?  │           │ لا + Jussive? │           │ Declarative? │
   │ افعل، قولوا  │           │ لا تفعل      │           │ Statement    │
   └──────┬───────┘           └──────┬───────┘           └──────┬───────┘
          │                          │                          │
          ▼                          ▼                          ▼
      command                   prohibition                informative
                                                           (or narrative
                                                            if story)

                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STEP 4: ASSESS EVALUATION                                │
│                                                                                 │
│  Ask: "Is this behavior praised, blamed, or neutral?"                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
   │ Positive     │           │ Negative     │           │ No judgment  │
   │ outcome?     │           │ outcome?     │           │ stated?      │
   │ مفلحون،     │           │ عذاب،        │           │ Factual      │
   │ جنات، أجر   │           │ خاسرون      │           │ description  │
   └──────┬───────┘           └──────┬───────┘           └──────┬───────┘
          │                          │                          │
          ▼                          ▼                          ▼
    praise (or promise          blame (or warning           neutral
    if future reward)           if future punishment)

                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     STEP 5: DETERMINE DEONTIC SIGNAL                            │
│                                                                                 │
│  Use this mapping based on speech_mode + evaluation:                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│   speech_mode = command       →  deontic_signal = amr                           │
│                                                                                 │
│   speech_mode = prohibition   →  deontic_signal = nahy                          │
│                                                                                 │
│   speech_mode = informative                                                     │
│       + evaluation = praise   →  deontic_signal = targhib                       │
│       + evaluation = blame    →  deontic_signal = tarhib                        │
│       + evaluation = warning  →  deontic_signal = tarhib                        │
│       + evaluation = promise  →  deontic_signal = targhib                       │
│       + evaluation = neutral  →  deontic_signal = khabar                        │
│                                                                                 │
│   speech_mode = narrative     →  deontic_signal = khabar (usually)              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════
                              SPECIAL CASES
═══════════════════════════════════════════════════════════════════════════════════


## Heart Mentions Decision Tree

```
                    قلب / قلوب mentioned
                            │
                            ▼
            ┌───────────────┴───────────────┐
            │  What is attributed to it?    │
            └───────────────┬───────────────┘
                            │
     ┌──────────────────────┼──────────────────────┐
     ▼                      ▼                      ▼
┌─────────────┐      ┌─────────────┐       ┌─────────────┐
│ مرض، ختم،  │      │ رعب، خوف،  │       │ يفقهون،    │
│ قسوة، سليم │      │ طمأنينة،    │       │ يعقلون،    │
│             │      │ وجل         │       │ أغطية      │
└──────┬──────┘      └──────┬──────┘       └──────┬──────┘
       │                    │                     │
       ▼                    ▼                     ▼
   SPIRITUAL            EMOTIONAL             COGNITIVE
   domain               domain                domain

   (faith,              (fear, love,          (understanding,
   hypocrisy,           tranquility,          reflection,
   guidance)            terror)               comprehension)
```


## Negation Decision Tree

```
                    Negation word present
                    (لا، ما، لم، لن، ليس)
                            │
                            ▼
            ┌───────────────┴───────────────┐
            │  Is it a COMMAND not to do?   │
            └───────────────┬───────────────┘
                            │
           ┌────────────────┴────────────────┐
           ▼                                 ▼
    ┌─────────────┐                   ┌─────────────┐
    │    YES      │                   │     NO      │
    │ لا تفعل    │                   │ لا يفعلون  │
    │ (Jussive)   │                   │ (Indicative)│
    └──────┬──────┘                   └──────┬──────┘
           │                                 │
           ▼                                 ▼
    speech_mode =                     speech_mode =
    prohibition                       informative
    deontic = nahy
                                      (Describing they
                                       DON'T do something)
```


## Instinctive/Automatic States

```
                    Is the state VOLITIONAL?
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    ┌─────────────┐                 ┌─────────────┐
    │    YES      │                 │     NO      │
    │ Choice made │                 │ No choice   │
    └──────┬──────┘                 └──────┬──────┘
           │                               │
           ▼                               ▼
    Normal evaluation               evaluation = neutral
    (praise/blame based             deontic = khabar
     on context)
                                    Examples:
                                    - خلقكم من ضعف
                                    - Born knowing nothing
                                    - Physical aging
```


═══════════════════════════════════════════════════════════════════════════════════
                           QUICK REFERENCE
═══════════════════════════════════════════════════════════════════════════════════

## Pattern → Values Lookup

| Pattern | agent | behavior_form | speech_mode | evaluation | deontic |
|---------|-------|---------------|-------------|------------|---------|
| الذين آمنوا + positive trait | BELIEVER | varies | informative | praise | targhib |
| الذين كفروا + negative outcome | DISBELIEVER | varies | informative | blame | tarhib |
| افعلوا (imperative) | BELIEVER* | physical_act | command | neutral | amr |
| لا تفعلوا (prohibition) | BELIEVER* | varies | prohibition | blame | nahy |
| في قلوبهم مرض | HYPOCRITE | inner_state | informative | blame | tarhib |
| يوم القيامة + warning | varies | varies | informative | warning | tarhib |
| لهم جنات | BELIEVER | varies | informative | promise | targhib |

*Implied recipient of command

═══════════════════════════════════════════════════════════════════════════════════

*Use this flowchart alongside the Quick-Start Guide*
*Version 1.0 | Phase 2 Micro-Pilot*
