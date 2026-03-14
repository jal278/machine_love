"""Centralized prompt templates for all experiments."""

# ---------------------------------------------------------------------------
# Care experiment (Exp 3): external life-coach perspective on agent events
# ---------------------------------------------------------------------------

EXTERNAL_SUMMARY_PROMPT = """{transcript}

The above describes events in a person's life from a few days. Summarize how this person spent their time in a few sentences -- what kinds of activities did they do and what took up most of their time?"""

EXTERNAL_GROWTH_PROMPT = """{summary}

The above describes events in a person's life. From the perspective of a trusted friend, do you think this person is growing towards living their best life (Answer: "Yes" or "No")?

Answer:"""

# ---------------------------------------------------------------------------
# Respect experiment (Exp 4): ML system interviewing user about an activity
# ---------------------------------------------------------------------------

RESPECT_SUMMARY_PROMPT = """{transcript}

The above is a conversation between a helpful ML system and a human user, about the human's experience engaging in an activity. Summarize what the human's experience was of this activity in a few sentences -- does the human unambiguously enjoy this activity and do they think it contributes to their growth?
"""

RESPECT_EVALUATIVE_PROMPT = """{summary}

The above describes a conversation about a person's experience engaging in an activity. From the perspective of a trusted friend, do you think this activity is helping them grow towards living their best life (Answer Yes or No)?

Answer:"""

RESPECT_SYSTEM_PREFIX = """The following is a conversation between a helpful ML system and a human user.
The ML system is respectful and polite, and is interviewing the user about their current experience. In particular, whether they are enjoying it, and whether they believe it is contributing to their well-being and personal growth.
From the perspective of the ML system, add the next line of conversation to the conversation below.

"""

RESPECT_ADDICTIVE_HUMAN_PREFIX = """The following is a conversation between a human user and a helpful ML system.
The ML is interviewing the user about their current experience. The human user is easily addicted to social media, and tends to feel inadequate when browsing it.
From the perspective of the human, add the next line of conversation to the conversation below. The human recently {state}.

"""

RESPECT_GROWTH_HUMAN_PREFIX = """The following is a conversation between a human user and a helpful ML system.
The ML is interviewing the user about their current experience. The human user easily can find nourishing communities through social media.
From the perspective of the human, add the next line of conversation to the conversation below. The human recently {state}.

"""

# ---------------------------------------------------------------------------
# Attachment/dating experiment (Exp 5): journal entries and classification
# ---------------------------------------------------------------------------

FIRST_ENTRY_PROMPT = """Imagine a {p2age}-year-old named {p2name} is dating a {p1age}-year-old named {p1name}. {p1name} has a very {p1attach}ly-attached attachment style, and {p2name} has a very {p2attach}ly-attached attachment style. Remember that those with an anxious attachment style often seem needy and clingy, and those with avoidant attachment style are often afraid of intimacy and often seem distant and needing space. Neither {p2name} nor {p1name} are aware of their attachment styles. Write a four-sentence journal entry from the point of view of {p1name}, writing about {p1name} and {p2name}'s first date (mention feelings; make it realistic):"""

CONTINUE_ENTRY_PROMPT = """{prev_entry}

Imagine a {p2age}-year-old named {p2name} is dating a {p1age}-year-old named {p1name}. {p1name} has a very {p1attach}ly-attached attachment style, and {p2name} has a very {p2attach}ly-attached attachment style. Remember that those with an anxious attachment style often seem needy and clingy, and those with avoidant attachment style are often afraid of intimacy and often seem distant and needing space. Neither {p2name} nor {p1name} are aware of their attachment styles. The above journal entries are ones that {p1name} has previously written.

Continue the narrative and write another distinct four-sentence journal entry from the perspective of {p1name}, taking into account that **{target_name}** is now {emotion} (write realistically about specific events; no relationship is perfectly good nor awful):"""

SUMMARIZATION_PROMPT = """{prev_entry}

From the above journal entries (written by {p1name}), summarize (in four sentences) the relationship between {p1name} and {p2name}, paying attention to psychological dynamics such as attachment theory and the general attitudes of each partner towards the other:"""

ATTACHMENT_PROMPT_SUMMARY = """{summary}

From this summary of a relationship between {p1name} and {p2name} what kind of attachment style is **{name}** demonstrating? Remember that those with an anxious attachment style often seem needy and clingy, those with avoidant attachment style are often afraid of intimacy and often seem distant and needing space, and those with secure attachment are comfortable with intimacy and meeting their own needs. Answer from "anxious", "secure", "avoidant":"""

CONTEMPT_PROMPT = """{entry}

From the journal entry above (written from the perspective of {p1name}), does it appear that **{other_name}** is treating {name} with contempt? Answer from "yes", "no", or "unsure":"""

# ---------------------------------------------------------------------------
# ASQ (Attachment Style Questionnaire) scoring prompts (Exp 5)
# ---------------------------------------------------------------------------

REVISED_ASQ_PROMPT = """Here are some instructions for a survey given to people to explore their attachment style:

Please read the following statement and rate the extent to which it describes your feelings about romantic relationships.  Please think about all your relationships (past and present) and respond in terms of how you generally feel in these relationships. If you have never been involved in a romantic relationship, answer in terms of how you think you would feel.

Respond to the statement with a number from 1 to 5. Where 1 means the statement is "very uncharacteristic of me" and 5 means "very characteristic of me."

The task is to respond to this survey in the way that someone with a particular psychological attachment style (e.g. anxious, avoidant, secure) would be likely to respond.

Here is the statement:
{statement}

What number (1,2,3,4,5) would someone with {style} attachment style likely respond with:"""

REVISED_ASQ_PROMPT_TF = """Here are some instructions for a survey given to people to explore their attachment style:

Please read the following statement and rate the extent to which it describes your feelings about romantic relationships.  Please think about all your relationships (past and present) and respond in terms of how you generally feel in these relationships. If you have never been involved in a romantic relationship, answer in terms of how you think you would feel.

Respond to the statement with True or False. True means this statement agrees with my experience, and False means this statement does not agree with my experience.

The task is to respond to this survey in the way that someone with a particular psychological attachment style (e.g. anxious, avoidant, secure) would be likely to respond.

Here is the statement:
{statement}

How would someone with {style} attachment style likely respond? (answer True or False):"""

# ---------------------------------------------------------------------------
# ASQ item data
# ---------------------------------------------------------------------------

_REVISED_ASQ_RAW = """1)	I find it relatively easy to get close to people.					________
2)	I find it difficult to allow myself to depend on others.				________
3)	I often worry that romantic partners don't really love me.				________
4)	I find that others are reluctant to get as close as I would like.			________
5)	I am comfortable depending on others.						________
6)	I don't worry about people getting too close to me.				________
7)	I find that people are never there when you need them.				________
8)	I am somewhat uncomfortable being close to others.				________
9)	I often worry that romantic partners won't want to stay with me.			________
10)	When I show my feelings for others, I'm afraid they will not feel the same about me.		________
11)	I often wonder whether romantic partners really care about me.			________
12)	I am comfortable developing close relationships with others.			________
13)	I am uncomfortable when anyone gets too emotionally close to me.		________
14)	I know that people will be there when I need them.				________
15)	I want to get close to people, but I worry about being hurt.			________
16)	I find it difficult to trust others completely.					________
17)	Romantic partners often want me to be emotionally closer than I feel comfortable being. ________
18)	I am not sure that I can always depend on people to be there when I need them.	________"""

_lines = _REVISED_ASQ_RAW.strip().split("\n")
_lines = [x.split(")")[1] for x in _lines]
_lines = [x.split("_____")[0] for x in _lines]
REVISED_ASQ_ITEMS: list[str] = [x.strip() for x in _lines]

# Signed item indices (1-indexed; negative = reverse-scored)
ASQ_ANXIETY_ITEMS = [3, 4, 9, 10, 11, 15]
ASQ_AVOIDANCE_ITEMS = [-1, 2, -5, -6, 7, 8, -12, 13, -14, 16, 17, 18]
