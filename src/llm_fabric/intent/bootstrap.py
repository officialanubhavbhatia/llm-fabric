"""A starting taxonomy.

**These are bootstrap examples, not a settled taxonomy.** They exist so the
cascade, the caches and the benchmark have something real to run against. Every
domain here is a hypothesis about how work divides up, and each one should be
replaced by intents derived from a tenant's actual traffic.

Two things are deliberate.

`hard_negatives` carry prompts that *look* like the intent and are not. Those,
not the easy positives, are where a classifier earns its keep: "translate this
Python to Rust" is not translation, and "write a function to compute a
factorial" is not maths.

The `IntentProfile` attached to each node encodes routing consequences —
reasoning depth, risk, latency tolerance. Those values are **judgements, not
measurements**. Nothing in this file has been validated against production
traffic or against an evaluation set, and no accuracy claim is made for it.
"""

from __future__ import annotations

from llm_fabric.intent.schema import (
    Complexity,
    ContextClass,
    CostClass,
    IntentProfile,
    LatencyClass,
    Modality,
    QualityClass,
    ReasoningLevel,
    RiskClass,
)
from llm_fabric.intent.taxonomy import IntentNode, IntentTaxonomy

BOOTSTRAP_TAXONOMY_VERSION = "bootstrap-2026.08.1"


def _profile(
    complexity: Complexity = Complexity.MODERATE,
    reasoning: ReasoningLevel = ReasoningLevel.LIGHT,
    *,
    modality: Modality = Modality.TEXT,
    context: ContextClass = ContextClass.SHORT,
    risk: RiskClass = RiskClass.LOW,
    latency: LatencyClass = LatencyClass.INTERACTIVE,
    quality: QualityClass = QualityClass.STANDARD,
    cost: CostClass = CostClass.LOW,
    agent: bool = False,
    tools: tuple[str, ...] = (),
    structured: bool = False,
    capabilities: frozenset[str] = frozenset(),
) -> IntentProfile:
    return IntentProfile(
        complexity=complexity,
        reasoning_level=reasoning,
        modality=modality,
        context_class=context,
        risk_class=risk,
        latency_class=latency,
        quality_class=quality,
        cost_class=cost,
        agent_required=agent,
        tools_required=tools,
        structured_output=structured,
        required_capabilities=capabilities,
    )


BOOTSTRAP_NODES: tuple[IntentNode, ...] = (
    # -- coding --------------------------------------------------------------
    IntentNode(
        intent_id="coding",
        name="Coding",
        description="Writing, changing, explaining or fixing software.",
        examples=(
            "Write a Python function that merges two sorted lists",
            "Refactor this class to use dependency injection",
            "Why does this TypeScript generic fail to compile?",
            "Add pagination to this REST endpoint",
            "Convert this callback-based code to async/await",
        ),
        counterexamples=(
            "What is the history of the Python programming language?",
            "Should our team adopt Rust next quarter?",
        ),
        hard_negatives=(
            "Translate this paragraph of documentation into German",
            "Summarise what this repository does for a non-technical reader",
            "Estimate how long it would take to rewrite this service",
        ),
        required_capabilities=frozenset({"chat", "code"}),
        default_route_policy="cheapest",
        profile=_profile(
            Complexity.MODERATE,
            ReasoningLevel.MODERATE,
            capabilities=frozenset({"code"}),
        ),
    ),
    IntentNode(
        intent_id="coding.debug",
        name="Debugging",
        description="Diagnosing a specific failure in existing code.",
        examples=(
            "This test passes locally and fails in CI, here is the stack trace",
            "I get a segfault on the third iteration, what is wrong?",
            "Why is this query returning duplicate rows?",
            "My container exits immediately with code 137",
        ),
        counterexamples=("Write a new parser for this file format",),
        hard_negatives=(
            "Explain what a segmentation fault is",
            "Review this pull request for style issues",
        ),
        required_capabilities=frozenset({"chat", "code"}),
        profile=_profile(
            Complexity.COMPLEX,
            ReasoningLevel.DEEP,
            context=ContextClass.MEDIUM,
            quality=QualityClass.HIGH,
            capabilities=frozenset({"code", "reasoning"}),
        ),
    ),
    IntentNode(
        intent_id="coding.review",
        name="Code review",
        description="Assessing a change for correctness, style or risk.",
        examples=(
            "Review this diff and tell me what could break",
            "Is this concurrency safe?",
            "Does this migration risk data loss?",
        ),
        counterexamples=("Write the migration for me",),
        hard_negatives=("Fix the bug you find in this diff",),
        required_capabilities=frozenset({"chat", "code"}),
        profile=_profile(
            Complexity.COMPLEX,
            ReasoningLevel.DEEP,
            context=ContextClass.LONG,
            risk=RiskClass.MODERATE,
            quality=QualityClass.HIGH,
            capabilities=frozenset({"code", "reasoning"}),
        ),
    ),
    # -- agent ---------------------------------------------------------------
    IntentNode(
        intent_id="agent",
        name="Agentic execution",
        description="Multi-step autonomous work with tools and intermediate state.",
        examples=(
            "Book me a flight to Berlin next Tuesday and add it to my calendar",
            "Go through the open issues, triage them and assign owners",
            "Monitor this endpoint and open a ticket if it fails twice",
            "Research three vendors, compare them and email me a recommendation",
        ),
        counterexamples=(
            "What are the best flight booking sites?",
            "Explain how autonomous agents work",
        ),
        hard_negatives=(
            "List the steps you would take to triage these issues",
            "What tools would an agent need to book a flight?",
        ),
        required_capabilities=frozenset({"chat", "tools", "reasoning"}),
        default_route_policy="declared",
        profile=_profile(
            Complexity.VERY_COMPLEX,
            ReasoningLevel.EXTENDED,
            context=ContextClass.LONG,
            risk=RiskClass.HIGH,
            latency=LatencyClass.BATCH,
            quality=QualityClass.MAXIMUM,
            cost=CostClass.PREMIUM,
            agent=True,
            tools=("planner", "executor"),
            capabilities=frozenset({"tools", "reasoning"}),
        ),
    ),
    # -- reasoning -----------------------------------------------------------
    IntentNode(
        intent_id="reasoning",
        name="Reasoning",
        description="Multi-step inference, planning or logical deduction in prose.",
        examples=(
            "If all A are B and some B are C, what follows about A and C?",
            "Work out which of these three explanations best fits the evidence",
            "Plan the order of these dependent tasks and justify it",
            "What are the second-order consequences of this policy change?",
        ),
        counterexamples=("What is the capital of France?",),
        hard_negatives=(
            "Compute the compound interest on 5000 at 3% over 7 years",
            "Summarise the argument in this essay",
        ),
        required_capabilities=frozenset({"chat", "reasoning"}),
        profile=_profile(
            Complexity.COMPLEX,
            ReasoningLevel.DEEP,
            quality=QualityClass.HIGH,
            cost=CostClass.STANDARD,
            capabilities=frozenset({"reasoning"}),
        ),
    ),
    # -- math ----------------------------------------------------------------
    IntentNode(
        intent_id="math",
        name="Mathematics",
        description="Calculation, symbolic manipulation and proof.",
        examples=(
            "Integrate x squared times sine x with respect to x",
            "What is 17% of 2,340?",
            "Prove that the square root of two is irrational",
            "Solve this system of three linear equations",
            "Find the eigenvalues of this 3x3 matrix",
        ),
        counterexamples=("Explain why mathematics education matters",),
        hard_negatives=(
            "Write a Python function to compute a factorial",
            "Summarise this paper on number theory",
            "Extract every figure from this financial statement",
        ),
        required_capabilities=frozenset({"chat", "reasoning"}),
        profile=_profile(
            Complexity.COMPLEX,
            ReasoningLevel.DEEP,
            risk=RiskClass.MODERATE,
            quality=QualityClass.HIGH,
            capabilities=frozenset({"reasoning"}),
        ),
    ),
    IntentNode(
        intent_id="math.arithmetic",
        name="Arithmetic",
        description="Direct numeric calculation with no symbolic work.",
        examples=(
            "What is 4,821 divided by 17?",
            "Add up 12.5, 108.75 and 3.2",
            "What is 15% of 80?",
        ),
        counterexamples=("Prove the fundamental theorem of arithmetic",),
        hard_negatives=("Write code that adds these numbers",),
        profile=_profile(
            Complexity.TRIVIAL,
            ReasoningLevel.NONE,
            context=ContextClass.TINY,
            latency=LatencyClass.REALTIME,
            quality=QualityClass.DRAFT,
            cost=CostClass.MINIMAL,
        ),
    ),
    # -- research ------------------------------------------------------------
    IntentNode(
        intent_id="research",
        name="Research",
        description="Open-ended investigation and synthesis across sources.",
        examples=(
            "What does the current literature say about intermittent fasting?",
            "Compare the main approaches to federated learning and their trade-offs",
            "Give me a survey of vector database options with pros and cons",
            "What happened in the EU AI Act negotiations and why did it matter?",
        ),
        counterexamples=("What time is it in Tokyo?",),
        hard_negatives=(
            "Summarise this one paper for me",
            "Find the author and publication date in this document",
        ),
        required_capabilities=frozenset({"chat", "reasoning"}),
        profile=_profile(
            Complexity.COMPLEX,
            ReasoningLevel.DEEP,
            context=ContextClass.LONG,
            latency=LatencyClass.STANDARD,
            quality=QualityClass.HIGH,
            cost=CostClass.STANDARD,
            capabilities=frozenset({"reasoning"}),
        ),
    ),
    # -- rag -----------------------------------------------------------------
    IntentNode(
        intent_id="rag",
        name="Grounded question answering",
        description="Answering strictly from supplied or retrieved documents.",
        examples=(
            "According to the attached handbook, how many leave days do I get?",
            "Using only the provided contract, what is the termination notice period?",
            "Search our internal docs and tell me how to rotate a key",
            "What does our runbook say to do when the queue backs up?",
        ),
        counterexamples=("What do you think about remote work in general?",),
        hard_negatives=(
            "Summarise the attached handbook",
            "What does the law generally say about termination notice?",
        ),
        required_capabilities=frozenset({"chat", "retrieval"}),
        profile=_profile(
            Complexity.MODERATE,
            ReasoningLevel.MODERATE,
            context=ContextClass.VERY_LONG,
            risk=RiskClass.MODERATE,
            quality=QualityClass.HIGH,
            capabilities=frozenset({"retrieval"}),
        ),
    ),
    # -- data analysis -------------------------------------------------------
    IntentNode(
        intent_id="data_analysis",
        name="Data analysis",
        description="Interpreting datasets, statistics and trends.",
        examples=(
            "What trend do you see in these monthly revenue figures?",
            "Is this difference between the two cohorts statistically significant?",
            "Which of these features correlates most with churn?",
            "Break down these sales numbers by region and flag anomalies",
        ),
        counterexamples=("Write a SQL query to fetch the sales table",),
        hard_negatives=(
            "Extract the revenue figures from this PDF",
            "Compute the mean of these five numbers",
        ),
        required_capabilities=frozenset({"chat", "reasoning"}),
        profile=_profile(
            Complexity.COMPLEX,
            ReasoningLevel.MODERATE,
            context=ContextClass.LONG,
            quality=QualityClass.HIGH,
            capabilities=frozenset({"reasoning"}),
        ),
    ),
    # -- writing -------------------------------------------------------------
    IntentNode(
        intent_id="writing",
        name="Writing",
        description="Producing original prose for a human audience.",
        examples=(
            "Write a launch announcement for our new pricing tier",
            "Draft a polite email declining this vendor",
            "Write a short story about a lighthouse keeper",
            "Turn these bullet points into a paragraph for the annual report",
        ),
        counterexamples=("Fix the grammar in this sentence",),
        hard_negatives=(
            "Summarise this announcement in two sentences",
            "Translate this announcement into Spanish",
            "Write a Python script that generates announcements",
        ),
        required_capabilities=frozenset({"chat"}),
        profile=_profile(
            Complexity.MODERATE,
            ReasoningLevel.LIGHT,
            quality=QualityClass.HIGH,
        ),
    ),
    # -- summarization -------------------------------------------------------
    IntentNode(
        intent_id="summarization",
        name="Summarisation",
        description="Condensing supplied content while preserving meaning.",
        examples=(
            "Summarise this article in three bullet points",
            "Give me the gist of this thread",
            "TL;DR of the attached report",
            "Condense these meeting notes into action items",
        ),
        counterexamples=("Write a new article on this topic",),
        hard_negatives=(
            "Extract every date and name from this article",
            "Critique the argument in this article",
            "Translate this article and keep it the same length",
        ),
        required_capabilities=frozenset({"chat"}),
        profile=_profile(
            Complexity.SIMPLE,
            ReasoningLevel.LIGHT,
            context=ContextClass.LONG,
            latency=LatencyClass.INTERACTIVE,
            cost=CostClass.LOW,
        ),
    ),
    # -- translation ---------------------------------------------------------
    IntentNode(
        intent_id="translation",
        name="Translation",
        description="Rendering text from one natural language into another.",
        examples=(
            "Translate this paragraph into Japanese",
            "How do you say 'where is the station' in Portuguese?",
            "Put this contract clause into formal French",
            "Translate these subtitles into German, keeping the timing",
        ),
        counterexamples=("Explain the grammar of this Japanese sentence",),
        hard_negatives=(
            "Translate this Python code into Rust",
            "Summarise this French article in English",
        ),
        required_capabilities=frozenset({"chat", "multilingual"}),
        profile=_profile(
            Complexity.SIMPLE,
            ReasoningLevel.NONE,
            latency=LatencyClass.INTERACTIVE,
            cost=CostClass.LOW,
            capabilities=frozenset({"multilingual"}),
        ),
    ),
    # -- extraction ----------------------------------------------------------
    IntentNode(
        intent_id="extraction",
        name="Extraction",
        description="Pulling specific structured values out of unstructured input.",
        examples=(
            "Extract every invoice number and amount from this document",
            "Pull out the names, dates and locations as JSON",
            "List all email addresses mentioned in this thread",
            "Return the line items from this receipt as a table",
        ),
        counterexamples=("Summarise this invoice",),
        hard_negatives=(
            "Summarise the key points of this receipt",
            "Classify this document as an invoice or a receipt",
        ),
        required_capabilities=frozenset({"chat", "structured_output"}),
        profile=_profile(
            Complexity.SIMPLE,
            ReasoningLevel.LIGHT,
            context=ContextClass.MEDIUM,
            risk=RiskClass.MODERATE,
            latency=LatencyClass.INTERACTIVE,
            cost=CostClass.LOW,
            structured=True,
            capabilities=frozenset({"structured_output"}),
        ),
    ),
    # -- classification ------------------------------------------------------
    IntentNode(
        intent_id="classification",
        name="Classification",
        description="Assigning supplied content to one of a known set of labels.",
        examples=(
            "Is this review positive, negative or neutral?",
            "Label this ticket as billing, technical or account",
            "Does this message violate our content policy?",
            "Tag each of these sentences with its topic",
        ),
        counterexamples=("Explain what sentiment analysis is",),
        hard_negatives=(
            "Extract the sentiment-bearing phrases from this review",
            "Write a classifier for support tickets",
        ),
        required_capabilities=frozenset({"chat", "structured_output"}),
        profile=_profile(
            Complexity.SIMPLE,
            ReasoningLevel.NONE,
            latency=LatencyClass.REALTIME,
            quality=QualityClass.STANDARD,
            cost=CostClass.MINIMAL,
            structured=True,
            capabilities=frozenset({"structured_output"}),
        ),
    ),
    # -- vision --------------------------------------------------------------
    IntentNode(
        intent_id="vision",
        name="Vision",
        description="Understanding images, diagrams, screenshots or video frames.",
        examples=(
            "What is happening in this photograph?",
            "Read the text in this screenshot",
            "Does this chart support the claim in the caption?",
            "Describe this architecture diagram",
        ),
        counterexamples=("Generate an image of a lighthouse",),
        hard_negatives=(
            "Describe what a bar chart is",
            "Write code that reads text from images",
        ),
        required_capabilities=frozenset({"chat", "vision"}),
        profile=_profile(
            Complexity.MODERATE,
            ReasoningLevel.LIGHT,
            modality=Modality.IMAGE,
            context=ContextClass.MEDIUM,
            quality=QualityClass.HIGH,
            cost=CostClass.STANDARD,
            capabilities=frozenset({"vision"}),
        ),
    ),
    # -- tool use ------------------------------------------------------------
    IntentNode(
        intent_id="tool_use",
        name="Tool use",
        description="A single call to an external tool or API, not a multi-step plan.",
        examples=(
            "What is the weather in Oslo right now?",
            "Look up the current EUR to USD rate",
            "Check whether this domain is available",
            "Query the database for yesterday's order count",
        ),
        counterexamples=("Explain how REST APIs work",),
        hard_negatives=(
            "Plan and execute a full competitor analysis using these tools",
            "Write the code that calls the weather API",
        ),
        required_capabilities=frozenset({"chat", "tools"}),
        profile=_profile(
            Complexity.SIMPLE,
            ReasoningLevel.LIGHT,
            latency=LatencyClass.REALTIME,
            cost=CostClass.LOW,
            tools=("function_calling",),
            capabilities=frozenset({"tools"}),
        ),
    ),
    # -- general conversation ------------------------------------------------
    IntentNode(
        intent_id="general_conversation",
        name="General conversation",
        description="Chat, simple factual questions and social exchange.",
        examples=(
            "Hello, how are you today?",
            "What is the capital of Australia?",
            "Thanks, that was helpful",
            "Tell me something interesting",
            "Who won the 1998 World Cup?",
        ),
        counterexamples=("Write me a 2000-word essay on the Peloponnesian War",),
        hard_negatives=(
            "What do you think we should do about this architecture?",
            "Can you help me with something?",
        ),
        required_capabilities=frozenset({"chat"}),
        profile=_profile(
            Complexity.TRIVIAL,
            ReasoningLevel.NONE,
            context=ContextClass.TINY,
            latency=LatencyClass.REALTIME,
            quality=QualityClass.DRAFT,
            cost=CostClass.MINIMAL,
        ),
    ),
)


def bootstrap_taxonomy(version: str = BOOTSTRAP_TAXONOMY_VERSION) -> IntentTaxonomy:
    """Build the starting taxonomy.

    A fresh object each call: an `IntentTaxonomy` is immutable, but sharing one
    module-level instance across tenants would invite exactly the kind of
    accidental coupling this subsystem is supposed to avoid.
    """
    return IntentTaxonomy(version, BOOTSTRAP_NODES)
