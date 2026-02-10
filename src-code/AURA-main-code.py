# ==========================Factories + Engines==========================
# AURA v7 — Neuro‑Symbolic Hybrid Brain
# Chunk 1 / 10
# Imports, Utilities, Embedding Service
# ==========================

import time
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterable, Tuple, Callable

import torch
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset


# ==========================
# Device & Embedding Model
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

print("Using device:", DEVICE)


# ==========================
# Utility Functions
# ==========================

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if text is None:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def contains_negation(text: str) -> bool:
    """Detect negation words in natural language."""
    neg_words = ["not", "never", "no", "doesnt", "isnt", "arent", "without", "cannot", "can't"]
    tokens = normalize_text(text).split()
    return any(w in tokens for w in neg_words)


def extract_prob_modifier(text: str) -> float:
    """
    Detect probabilistic language and convert to confidence multipliers.
    Examples:
        "always" -> 1.0
        "usually" -> 0.8
        "often" -> 0.7
        "sometimes" -> 0.5
        "rarely" -> 0.3
        "never" -> 0.0
    """
    t = normalize_text(text)
    if "always" in t:
        return 1.0
    if "usually" in t:
        return 0.8
    if "often" in t:
        return 0.7
    if "sometimes" in t:
        return 0.5
    if "rarely" in t:
        return 0.3
    if "never" in t:
        return 0.0
    return 1.0  # default


def is_variable(token: Optional[str]) -> bool:
    """Variables start with ? (e.g., ?x, ?y)."""
    return isinstance(token, str) and token.startswith("?") and len(token) > 1


# ==========================
# Embedding Service
# ==========================

class EmbeddingService:
    """Wrapper around SentenceTransformer for consistent encoding."""
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: torch.device = DEVICE):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, text: str) -> torch.Tensor:
        return self.model.encode(text, convert_to_tensor=True)

    def encode_batch(self, texts: Iterable[str]) -> torch.Tensor:
        return self.model.encode(list(texts), convert_to_tensor=True)


# Global embedding service instance
embedding_service = EmbeddingService()
# ==========================
# AURA v7 — Neuro‑Symbolic Hybrid Brain
# Chunk 2 / 10
# Core Data Structures
# ==========================

@dataclass(frozen=True)
class Fact:
    """
    A structured fact with:
    - subject
    - predicate
    - object
    - confidence (0–1)
    - polarity (+1 = positive, -1 = negated)
    - source (manual, induced, nl_input, dataset, etc.)
    """
    subject: str
    predicate: str
    obj: Optional[str] = None
    confidence: float = 1.0
    polarity: int = 1
    verified: bool = True
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    embedding: torch.Tensor = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        # Normalize fields
        object.__setattr__(self, "subject", normalize_text(self.subject))
        object.__setattr__(self, "predicate", normalize_text(self.predicate))
        if self.obj is not None:
            object.__setattr__(self, "obj", normalize_text(self.obj))

        # Create canonical text for embedding
        canonical = self.to_text(include_polarity=False)
        emb = embedding_service.encode(canonical)
        object.__setattr__(self, "embedding", emb)

    def to_text(self, include_polarity: bool = True) -> str:
        """Return human-readable representation."""
        base = f"{self.subject} {self.predicate}"
        if self.obj:
            base += f" {self.obj}"
        if include_polarity and self.polarity == -1:
            base = "NOT " + base
        return base

    def key(self) -> str:
        """Unique key for memory storage."""
        pol = "neg" if self.polarity == -1 else "pos"
        return f"{pol}:{self.to_text(include_polarity=False)}"


@dataclass
class RulePattern:
    """
    A pattern used in rules, supporting variables (?x, ?y).
    Includes polarity for negation-aware reasoning.
    """
    subject: str
    predicate: str
    obj: Optional[str] = None
    polarity: int = 1

    def normalized(self) -> "RulePattern":
        return RulePattern(
            subject=normalize_text(self.subject),
            predicate=normalize_text(self.predicate),
            obj=normalize_text(self.obj) if self.obj is not None else None,
            polarity=self.polarity,
        )


@dataclass
class Rule:
    """
    A rule with:
    - name
    - list of condition patterns
    - conclusion pattern
    - confidence
    - source (manual, induced)
    """
    name: str
    conditions: List[RulePattern]
    conclusion: RulePattern
    confidence: float = 1.0
    source: str = "manual"

    def normalized(self) -> "Rule":
        return Rule(
            name=self.name,
            conditions=[c.normalized() for c in self.conditions],
            conclusion=self.conclusion.normalized(),
            confidence=self.confidence,
            source=self.source,
        )


@dataclass
class ReasoningEvent:
    """
    A log entry for meta-memory:
    - which engine produced it
    - message
    - confidence
    - timestamp
    """
    engine: str
    message: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
# ==========================
# AURA v7 — Neuro‑Symbolic Hybrid Brain
# Chunk 3 / 10
# Retrieval Index + Memory Systems
# ==========================

# ==========================
# Retrieval Index
# ==========================

class RetrievalIndex:
    """
    Stores embeddings for fast similarity search.
    Used by semantic memory and analogy engine.
    """
    def __init__(self):
        self.embeddings: List[torch.Tensor] = []
        self.items: List[Any] = []

    def add(self, embedding: torch.Tensor, item: Any):
        self.embeddings.append(embedding)
        self.items.append(item)

    def search(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Tuple[Any, float]]:
        if not self.embeddings:
            return []
        mat = torch.stack(self.embeddings)
        sims = util.cos_sim(query_embedding, mat)[0]
        k = min(top_k, len(sims))
        topk = torch.topk(sims, k=k)
        results = []
        for idx in topk.indices:
            i = idx.item()
            results.append((self.items[i], sims[i].item()))
        return results


# ==========================
# Sensory Memory
# ==========================

class SensoryMemory:
    """
    Stores raw text entries (e.g., dataset items, Wikipedia text).
    Used as fallback when structured reasoning fails.
    """
    def __init__(self):
        self.entries: List[str] = []
        self.entry_embeddings: List[torch.Tensor] = []

    def add_entry(self, text: str, embedding: Optional[torch.Tensor] = None):
        self.entries.append(text)
        if embedding is None:
            embedding = embedding_service.encode(normalize_text(text))
        self.entry_embeddings.append(embedding)


# ==========================
# Working Memory
# ==========================

class WorkingMemory:
    """
    Stores active facts used for reasoning.
    """
    def __init__(self):
        self.facts: Dict[str, Fact] = {}

    def add_fact(self, fact: Fact):
        self.facts[fact.key()] = fact

    def has_fact(self, fact: Fact) -> bool:
        return fact.key() in self.facts

    def all_facts(self) -> List[Fact]:
        return list(self.facts.values())


# ==========================
# Semantic Memory
# ==========================

class SemanticMemory:
    """
    Stores long-term structured knowledge.
    Uses a graph + retrieval index.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.index = RetrievalIndex()
        self.fact_map: Dict[str, Fact] = {}

    def add_fact(self, fact: Fact):
        key = fact.key()
        # If fact exists, keep the one with higher confidence
        if key in self.fact_map:
            existing = self.fact_map[key]
            if fact.confidence > existing.confidence:
                self.fact_map[key] = fact
            return

        self.fact_map[key] = fact
        self.graph.add_node(key, data=fact)
        self.index.add(fact.embedding, fact)

    def add_relation(self, source: Fact, target: Fact, relation: str, weight: float = 1.0):
        self.graph.add_edge(source.key(), target.key(), relation=relation, weight=weight)

    def search_fact(self, query_embedding: torch.Tensor, threshold: float = 0.6) -> Optional[Fact]:
        results = self.index.search(query_embedding, top_k=5)
        best = [(f, score) for f, score in results if score >= threshold]
        if not best:
            return None
        best.sort(key=lambda x: x[1], reverse=True)
        return best[0][0]


# ==========================
# Episodic Memory
# ==========================

@dataclass
class Episode:
    description: str
    facts: List[Fact]
    timestamp: float = field(default_factory=time.time)


class EpisodicMemory:
    """
    Stores episodes: groups of facts tied to a specific event or dataset item.
    """
    def __init__(self):
        self.episodes: List[Episode] = []

    def add_episode(self, description: str, facts: List[Fact]):
        self.episodes.append(Episode(description=description, facts=facts))


# ==========================
# Procedural Memory
# ==========================

class ProceduralMemory:
    """
    Stores learned procedures (functions).
    """
    def __init__(self):
        self.procedures: Dict[str, Callable] = {}

    def add_procedure(self, name: str, func: Callable):
        self.procedures[name] = func

    def get_procedure(self, name: str) -> Optional[Callable]:
        return self.procedures.get(name)


# ==========================
# Meta-Memory
# ==========================

class MetaMemory:
    """
    Tracks:
    - reasoning events
    - contradictions
    - engine reliability scores
    """
    def __init__(self):
        self.contradictions: List[str] = []
        self.events: List[ReasoningEvent] = []
        self.engine_scores: Dict[str, float] = {}  # reliability scores

    def log(self, engine: str, message: str, confidence: float):
        self.events.append(ReasoningEvent(engine=engine, message=message, confidence=confidence))
        # Update engine reliability
        self.engine_scores[engine] = self.engine_scores.get(engine, 0.5) * 0.9 + confidence * 0.1

    def add_contradiction(self, fact1: Fact, fact2: Fact):
        msg = f"CONTRADICTION: '{fact1.to_text()}' conflicts with '{fact2.to_text()}'"
        self.contradictions.append(msg)
        self.events.append(ReasoningEvent(engine="TMS", message=msg, confidence=0.0))

    def recent_trace(self, n: int = 20) -> List[ReasoningEvent]:
        return self.events[-n:]
# ==========================
# AURA v7 — Neuro‑Symbolic Hybrid Brain
# Chunk 4 / 10
# Unification Engine
# ==========================

from abc import ABC, abstractmethod

class ReasoningEngineBase(ABC):
    """Abstract base class for all reasoning engines."""
    name: str

    @abstractmethod
    def reason_forward(self, engine: "CognitiveEngine") -> Tuple[List["Fact"], List[str], float]:
        pass

    @abstractmethod
    def reason_backward(self, engine: "CognitiveEngine", goal: "RulePattern") -> Tuple[List["Fact"], List[str], float]:
        pass

def unify_token(pattern: Optional[str], value: Optional[str], subst: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Unify a single token:
    - If pattern is a variable (?x), bind it.
    - If pattern is a constant, it must match value.
    """
    if pattern is None and value is None:
        return subst
    if pattern is None or value is None:
        return None

    pattern = normalize_text(pattern)
    value = normalize_text(value)

    # Variable case
    if is_variable(pattern):
        var = pattern
        if var in subst:
            # Already bound → must match
            return subst if subst[var] == value else None
        # Bind new variable
        new_subst = dict(subst)
        new_subst[var] = value
        return new_subst

    # Constant case
    if pattern == value:
        return subst

    return None


def unify_fact(pattern: "RulePattern", fact: "Fact", subst: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Unify a rule pattern with a fact.
    Includes polarity matching.
    """
    # Polarity must match
    if pattern.polarity != fact.polarity:
        return None

    # Subject
    subst1 = unify_token(pattern.subject, fact.subject, subst)
    if subst1 is None:
        return None

    # Predicate
    subst2 = unify_token(pattern.predicate, fact.predicate, subst1)
    if subst2 is None:
        return None

    # Object
    subst3 = unify_token(pattern.obj, fact.obj, subst2)
    return subst3


def apply_substitution(pattern: "RulePattern", subst: Dict[str, str]) -> "RulePattern":
    """
    Apply variable bindings to a rule pattern.
    """
    def apply_token(token: Optional[str]) -> Optional[str]:
        if token is None:
            return None
        token = normalize_text(token)
        if is_variable(token) and token in subst:
            return subst[token]
        return token

    return RulePattern(
        subject=apply_token(pattern.subject),
        predicate=apply_token(pattern.predicate),
        obj=apply_token(pattern.obj),
        polarity=pattern.polarity,
    )
# ==========================
# AURA v7 — Neuro‑Symbolic Hybrid Brain
# Chunk 5 / 10
# Deductive Engine (Forward + Backward)
# ==========================

class DeductiveEngine(ReasoningEngineBase):
    name = "deductive"

    # ---------------------------------------------------------
    # Forward Chaining
    # ---------------------------------------------------------
    def reason_forward(self, engine: "CognitiveEngine") -> Tuple[List[Fact], List[str], float]:
        new_facts: List[Fact] = []
        trace: List[str] = []
        avg_conf = 0.0
        count = 0

        facts = engine.working.all_facts()
        rules = [r.normalized() for r in engine.rules]

        for rule in rules:
            matches = self._match_rule(rule, facts)

            for subst, cond_facts in matches:
                concl_pattern = apply_substitution(rule.conclusion, subst)

                # Polarity propagation
                polarity = concl_pattern.polarity

                # Confidence propagation
                conf = self._propagate_confidence(rule, cond_facts)

                concl_fact = Fact(
                    subject=concl_pattern.subject,
                    predicate=concl_pattern.predicate,
                    obj=concl_pattern.obj,
                    polarity=polarity,
                    confidence=conf,
                    verified=True,
                    source=f"rule:{rule.name}",
                )

                # Contradiction detection
                self._check_contradictions(engine, concl_fact)

                if not engine.working.has_fact(concl_fact):
                    new_facts.append(concl_fact)
                    used = ", ".join(f.to_text() for f in cond_facts)
                    msg = (
                        f"[Deduction] {rule.name} with {used} "
                        f"-> {concl_fact.to_text()} (conf={concl_fact.confidence:.2f})"
                    )
                    trace.append(msg)
                    avg_conf += concl_fact.confidence
                    count += 1

        if count > 0:
            avg_conf /= count
        else:
            avg_conf = 0.0

        return new_facts, trace, avg_conf

    # ---------------------------------------------------------
    # Backward Chaining
    # ---------------------------------------------------------
    def reason_backward(self, engine: "CognitiveEngine", goal: RulePattern) -> Tuple[List[Fact], List[str], float]:
        trace: List[str] = []
        proven_facts: List[Fact] = []
        avg_conf = 0.0
        count = 0

        # 1. Check if goal already exists in working memory
        for f in engine.working.all_facts():
            if unify_fact(goal, f, {}) is not None:
                proven_facts.append(f)
                trace.append(f"[Backward-Deduction] Goal already known: {f.to_text()}")
                avg_conf += f.confidence
                count += 1
                return proven_facts, trace, avg_conf / max(count, 1)

        # 2. Try to prove goal using rules
        rules = [r.normalized() for r in engine.rules]

        for rule in rules:
            subst = unify_fact(rule.conclusion, Fact(goal.subject, goal.predicate, goal.obj, polarity=goal.polarity), {})
            if subst is None:
                continue

            # Try to prove all conditions
            all_proven = True
            cond_facts: List[Fact] = []

            for cond in rule.conditions:
                cond_goal = apply_substitution(cond, subst)
                pf, pt, pc = self.reason_backward(engine, cond_goal)
                trace.extend(pt)

                if not pf:
                    all_proven = False
                    break

                cond_facts.extend(pf)
                avg_conf += pc
                count += 1

            if all_proven:
                concl_pattern = apply_substitution(rule.conclusion, subst)
                concl_fact = Fact(
                    subject=concl_pattern.subject,
                    predicate=concl_pattern.predicate,
                    obj=concl_pattern.obj,
                    polarity=concl_pattern.polarity,
                    confidence=self._propagate_confidence(rule, cond_facts),
                    verified=True,
                    source=f"rule:{rule.name}",
                )

                proven_facts.append(concl_fact)
                trace.append(f"[Backward-Deduction] Proved {concl_fact.to_text()} via {rule.name}")
                avg_conf += concl_fact.confidence
                count += 1
                break

        if count > 0:
            avg_conf /= count
        else:
            avg_conf = 0.0

        return proven_facts, trace, avg_conf

    # ---------------------------------------------------------
    # Rule Matching
    # ---------------------------------------------------------
    def _match_rule(self, rule: Rule, facts: List[Fact]) -> List[Tuple[Dict[str, str], List[Fact]]]:
        results: List[Tuple[Dict[str, str], List[Fact]]] = []

        def backtrack(i: int, subst: Dict[str, str], chosen: List[Fact]):
            if i == len(rule.conditions):
                results.append((subst, chosen.copy()))
                return

            pattern = rule.conditions[i]

            for fact in facts:
                new_subst = unify_fact(pattern, fact, subst)
                if new_subst is not None and fact not in chosen:
                    chosen.append(fact)
                    backtrack(i + 1, new_subst, chosen)
                    chosen.pop()

        backtrack(0, {}, [])
        return results

    # ---------------------------------------------------------
    # Confidence Propagation
    # ---------------------------------------------------------
    def _propagate_confidence(self, rule: Rule, cond_facts: List[Fact]) -> float:
        conf = rule.confidence
        for f in cond_facts:
            conf *= f.confidence
        return max(min(conf, 1.0), 0.0)

    # ---------------------------------------------------------
    # Contradiction Detection
    # ---------------------------------------------------------
    def _check_contradictions(self, engine: "CognitiveEngine", new_fact: Fact):
        """
        If a fact with opposite polarity exists, log contradiction.
        """
        opposite_key = ("neg:" if new_fact.polarity == 1 else "pos:") + new_fact.to_text(include_polarity=False)

        if opposite_key in engine.semantic.fact_map:
            engine.meta.add_contradiction(new_fact, engine.semantic.fact_map[opposite_key])
# ==========================
# AURA v7 — Neuro‑Symbolic Hybrid Brain
# Chunk 6 / 10
# Inductive Engine + Analogical Engine
# ==========================

class InductiveEngine(ReasoningEngineBase):
    """
    Learns new rules from repeated relational patterns.
    Example:
        If we see:
            fire is hot
            touching_fire causes burn
            stove is hot
            touching_stove causes burn
        → Induce rule: hot things burn
    """
    name = "inductive"

    def reason_forward(self, engine: "CognitiveEngine") -> Tuple[List[Fact], List[str], float]:
        trace: List[str] = []
        new_facts: List[Fact] = []
        avg_conf = 0.0
        count = 0

        facts = engine.working.all_facts()

        # Collect patterns
        hot_map = {}
        burn_map = {}

        for f in facts:
            if f.predicate == "is" and f.obj == "hot" and f.polarity == 1:
                hot_map[f.subject] = f
            if f.predicate == "causes" and f.obj == "burn" and f.subject.startswith("touching_") and f.polarity == 1:
                x = f.subject.replace("touching_", "")
                burn_map[x] = f

        # Look for repeated pattern
        common = set(hot_map.keys()) & set(burn_map.keys())

        if len(common) >= 2:
            # Induce rule
            rule = Rule(
                name="induced_hot_things_burn_rule",
                conditions=[RulePattern(subject="?x", predicate="is", obj="hot", polarity=1)],
                conclusion=RulePattern(subject="touching_?x", predicate="causes", obj="burn", polarity=1),
                confidence=0.8,
                source="induced",
            )

            if rule.name not in [r.name for r in engine.rules]:
                engine.rules.append(rule)
                msg = "[Induction] Learned rule: if ?x is hot → touching_?x causes burn"
                trace.append(msg)
                avg_conf = 0.8
                count = 1

        return new_facts, trace, avg_conf

    def reason_backward(self, engine: "CognitiveEngine", goal: RulePattern):
        # Induction is forward-only
        return [], [], 0.0


# ==========================
# Analogical Engine
# ==========================

class AnalogicalEngine(ReasoningEngineBase):
    """
    Structural analogy + embeddings.
    Example:
        fire is hot
        touching_fire causes burn
        stove is hot
        → touching_stove causes burn (by analogy)
    """
    name = "analogical"

    def reason_forward(self, engine: "CognitiveEngine") -> Tuple[List[Fact], List[str], float]:
        trace = []
        new_facts = []
        avg_conf = 0.0
        count = 0

        facts = engine.working.all_facts()

        hot_subjects = [f for f in facts if f.predicate == "is" and f.obj == "hot" and f.polarity == 1]
        burn_facts = [f for f in facts if f.predicate == "causes" and f.obj == "burn" and f.subject.startswith("touching_")]

        for hf in hot_subjects:
            for bf in burn_facts:
                x = bf.subject.replace("touching_", "")
                if x == hf.subject:
                    # Find analogous subjects
                    for hf2 in hot_subjects:
                        if hf2.subject == hf.subject:
                            continue

                        sim = util.cos_sim(hf.embedding, hf2.embedding).item()
                        if sim > 0.6:
                            concl = Fact(
                                subject=f"touching_{hf2.subject}",
                                predicate="causes",
                                obj="burn",
                                polarity=1,
                                confidence=0.7 * sim,
                                verified=False,
                                source="analogical",
                            )

                            if not engine.working.has_fact(concl):
                                new_facts.append(concl)
                                msg = (
                                    f"[Analogy] From {hf.subject}~{hf2.subject} and {bf.to_text()} "
                                    f"→ {concl.to_text()} (sim={sim:.2f})"
                                )
                                trace.append(msg)
                                avg_conf += concl.confidence
                                count += 1

        if count > 0:
            avg_conf /= count
        else:
            avg_conf = 0.0

        return new_facts, trace, avg_conf

    def reason_backward(self, engine: "CognitiveEngine", goal: RulePattern):
        # Analogy is forward-only
        return [], [], 0.0
# ==========================
# AURA v7 — Neuro‑Symbolic Hybrid Brain
# Chunk 7 / 10
# Counterfactual Engine + Meta‑Controller + CognitiveEngine
# ==========================

class CounterfactualEngine(ReasoningEngineBase):
    """
    Hybrid counterfactual reasoning:
    - Symbolic intervention: replace a fact and re-run reasoning
    - Causal-ish graph reasoning via semantic memory
    """
    name = "counterfactual"

    def reason_forward(self, engine: "CognitiveEngine"):
        # Counterfactuals are not automatically generated
        return [], [], 0.0

    def reason_backward(self, engine: "CognitiveEngine", goal: RulePattern):
        # Counterfactuals require explicit user request
        return [], [], 0.0

    def simulate(self, engine: "CognitiveEngine", intervention_fact: Fact) -> Dict[str, Any]:
        """
        Perform a symbolic intervention:
        - Temporarily add the fact
        - Re-run forward reasoning
        - Observe differences
        """
        temp_engine = engine.clone()

        temp_engine.add_fact(intervention_fact)
        temp_engine.reason_forward_until_fixpoint(max_iterations=3)

        return {
            "intervention": intervention_fact.to_text(),
            "new_facts": [f.to_text() for f in temp_engine.working.all_facts()],
        }


# ==========================
# Meta-Controller
# ==========================

class MetaController:
    """
    Chooses which reasoning engines to trust based on:
    - Past performance (engine_scores)
    - Confidence of recent outputs
    """
    def __init__(self, meta_memory: MetaMemory):
        self.meta = meta_memory

    def choose_engines_forward(self, engines: List[ReasoningEngineBase]) -> List[ReasoningEngineBase]:
        scored = []
        for e in engines:
            score = self.meta.engine_scores.get(e.name, 0.5)
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored]

    def choose_engines_backward(self, engines: List[ReasoningEngineBase]) -> List[ReasoningEngineBase]:
        return self.choose_engines_forward(engines)


# ==========================
# Cognitive Engine (Core Brain)
# ==========================

class CognitiveEngine:
    """
    The central orchestrator:
    - Holds all memory systems
    - Holds all reasoning engines
    - Runs forward/backward reasoning
    - Handles contradictions
    """
    def __init__(self):
        # Memory systems
        self.sensory = SensoryMemory()
        self.working = WorkingMemory()
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory()
        self.procedural = ProceduralMemory()
        self.meta = MetaMemory()

        # Reasoning engines
        self.engines: List[ReasoningEngineBase] = [
            DeductiveEngine(),
            InductiveEngine(),
            AnalogicalEngine(),
            CounterfactualEngine(),
        ]

        # Meta-controller
        self.meta_controller = MetaController(self.meta)

        # Rule store
        self.rules: List[Rule] = []

    # -------------------------
    # Fact & Rule Management
    # -------------------------

    def add_fact(self, fact: Fact):
        """
        Add fact to working + semantic memory.
        Check for contradictions.
        """
        # Check for contradictions
        opposite_key = ("neg:" if fact.polarity == 1 else "pos:") + fact.to_text(include_polarity=False)
        if opposite_key in self.semantic.fact_map:
            self.meta.add_contradiction(fact, self.semantic.fact_map[opposite_key])

        self.working.add_fact(fact)
        self.semantic.add_fact(fact)

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    # -------------------------
    # Forward Reasoning Loop
    # -------------------------

    def reason_forward_until_fixpoint(self, max_iterations: int = 5):
        """
        Run forward reasoning until no new facts are produced.
        """
        for _ in range(max_iterations):
            any_new = False

            ordered_engines = self.meta_controller.choose_engines_forward(self.engines)

            for engine in ordered_engines:
                new_facts, trace, avg_conf = engine.reason_forward(self)

                if new_facts:
                    any_new = True
                    for f in new_facts:
                        self.add_fact(f)

                    for t in trace:
                        self.meta.log(engine.name, t, avg_conf if avg_conf > 0 else 0.5)

            if not any_new:
                break

    # -------------------------
    # Backward Reasoning
    # -------------------------

    def reason_backward(self, goal: RulePattern) -> Tuple[List[Fact], List[str], float]:
        ordered_engines = self.meta_controller.choose_engines_backward(self.engines)

        best_facts = []
        best_trace = []
        best_conf = 0.0

        for engine in ordered_engines:
            facts, trace, conf = engine.reason_backward(self, goal)
            if facts and conf > best_conf:
                best_facts = facts
                best_trace = trace
                best_conf = conf

        return best_facts, best_trace, best_conf

    # -------------------------
    # Cloning (for counterfactuals)
    # -------------------------

    def clone(self) -> "CognitiveEngine":
        """
        Create a deep-ish clone of the engine for counterfactual simulation.
        """
        new = CognitiveEngine()

        # Copy facts
        for f in self.working.all_facts():
            new.add_fact(f)

        # Copy rules
        for r in self.rules:
            new.add_rule(r)

        return new
# ==========================
# AURA v7 — Neuro‑Symbolic Hybrid Brain
# Chunk 8 / 10
# Natural Language Parser (Hybrid)
# ==========================

class NLParser:
    """
    Hybrid natural language parser:
    - Rule-based parsing for simple patterns
    - Embedding-based fallback
    - Negation detection
    - Probabilistic modifier extraction
    """

    def __init__(self, engine: CognitiveEngine):
        self.engine = engine

    # ---------------------------------------------------------
    # Assertion Parsing (creates Facts)
    # ---------------------------------------------------------
    def parse_assertion(self, text: str) -> Optional[Fact]:
        """
        Convert natural language assertions into structured Facts.
        Handles:
            - "X is Y"
            - "X causes Y"
            - Negation ("not", "never", etc.)
            - Probabilistic modifiers ("usually", "rarely")
        """
        t = normalize_text(text)
        neg = contains_negation(t)
        polarity = -1 if neg else 1
        prob = extract_prob_modifier(t)

        # Remove negation words for cleaner parsing
        t_clean = re.sub(r"\bnot\b|\bnever\b|\bdoesnt\b|\bisnt\b|\barent\b", "", t).strip()

        # Pattern: "x is y"
        m = re.match(r"(.+?) is (.+)", t_clean)
        if m:
            subj = m.group(1).strip()
            obj = m.group(2).strip()
            return Fact(
                subject=subj,
                predicate="is",
                obj=obj,
                polarity=polarity,
                confidence=prob,
                source="nl_input",
            )

        # Pattern: "x causes y"
        m = re.match(r"(.+?) causes (.+)", t_clean)
        if m:
            subj = m.group(1).strip()
            obj = m.group(2).strip()
            return Fact(
                subject=subj,
                predicate="causes",
                obj=obj,
                polarity=polarity,
                confidence=prob,
                source="nl_input",
            )

        # Fallback: treat as descriptive fact
        return Fact(
            subject=t_clean,
            predicate="describes",
            obj=None,
            polarity=polarity,
            confidence=prob,
            source="nl_input",
        )

    # ---------------------------------------------------------
    # Query Parsing (creates RulePattern)
    # ---------------------------------------------------------
    def parse_query_to_goal(self, text: str) -> RulePattern:
        """
        Convert natural language questions into RulePatterns.
        Handles:
            - "Is X Y?"
            - "Does X cause Y?"
            - Negation ("not", "never")
        """
        t = normalize_text(text)
        neg = contains_negation(t)
        polarity = -1 if neg else 1

        # Remove negation words for cleaner parsing
        t_clean = re.sub(r"\bnot\b|\bnever\b|\bdoesnt\b|\bisnt\b|\barent\b", "", t).strip()

        # Pattern: "is x y?"
        m = re.match(r"is (.+?) (.+)\??", t_clean)
        if m:
            subj = m.group(1).strip()
            obj = m.group(2).strip()
            return RulePattern(
                subject=subj,
                predicate="is",
                obj=obj,
                polarity=polarity,
            )

        # Pattern: "does x cause y?"
        m = re.match(r"does (.+?) cause (.+)\??", t_clean)
        if m:
            subj = m.group(1).strip()
            obj = m.group(2).strip()
            return RulePattern(
                subject=subj,
                predicate="causes",
                obj=obj,
                polarity=polarity,
            )

        # Fallback
        return RulePattern(
            subject=t_clean,
            predicate="describes",
            obj=None,
            polarity=polarity,
        )
# ==========================
# AURA v7 — Neuro‑Symbolic Hybrid Brain
# Chunk 9 / 10
# Dataset Ingestion (Omniscience)
# ==========================

def ingest_omniscience(engine: CognitiveEngine, max_items: int = 200):
    """
    Ingests the ArtificialAnalysis/AA-Omniscience-Public dataset
    into sensory, semantic, and episodic memory.

    Each dataset item typically contains:
        - "input": natural language prompt
        - "output": natural language answer
        - "metadata": optional
    """

    print("Loading Omniscience dataset...")
    ds = load_dataset("ArtificialAnalysis/AA-Omniscience-Public")
    data = ds["train"]

    parser = NLParser(engine)
    count = 0

    for item in data:
        text_in = item.get("input", "")
        text_out = item.get("output", "")

        if not text_in and not text_out:
            continue

        # -------------------------
        # Sensory Memory
        # -------------------------
        combined_text = f"{text_in} -> {text_out}"
        normalized = normalize_text(combined_text)
        emb = embedding_service.encode(normalized)
        engine.sensory.add_entry(combined_text, embedding=emb)

        # -------------------------
        # Structured Fact Extraction
        # -------------------------
        fact_in = parser.parse_assertion(text_in) if text_in else None
        fact_out = parser.parse_assertion(text_out) if text_out else None

        # Add to semantic + working memory
        local_facts = []
        if fact_in:
            engine.add_fact(fact_in)
            local_facts.append(fact_in)
        if fact_out:
            engine.add_fact(fact_out)
            local_facts.append(fact_out)

        # -------------------------
        # Episodic Memory
        # -------------------------
        if local_facts:
            engine.episodic.add_episode(
                description="omniscience_item",
                facts=local_facts
            )

        count += 1
        if count >= max_items:
            break

    print(f"Ingested {count} items from AA-Omniscience-Public dataset.")
# ==========================
# AURA v7 — Neuro‑Symbolic Hybrid Brain
# Chunk 10 / 10
# AURA Interface + main()
# ==========================

class AURAInterface:
    """
    High-level interface for interacting with AURA:
    - assert_fact(text)
    - query(text)
    """

    def __init__(self, engine: CognitiveEngine):
        self.engine = engine
        self.parser = NLParser(engine)

    @staticmethod
    def _evidence_text(fact: Fact) -> str:
        """Return a user-facing evidence phrase derived from a fact."""
        return fact.to_text(include_polarity=True)

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------
    def assert_fact(self, text: str) -> Fact:
        fact = self.parser.parse_assertion(text)
        self.engine.add_fact(fact)
        return fact

    # ---------------------------------------------------------
    # Queries
    # ---------------------------------------------------------
    def query(self, text: str) -> Dict[str, Any]:
        normalized = normalize_text(text)
        query_embedding = embedding_service.encode(normalized)

        # Parse into structured goal
        goal = self.parser.parse_query_to_goal(text)

        # If negated, require explicit proof
        if goal.polarity == -1:
            facts, trace, conf = self.engine.reason_backward(goal)
            if facts:
                best = max(facts, key=lambda f: f.confidence)
                return {
                    "response": self._evidence_text(best),
                    "explanation": f"Proved negated fact: '{best.to_text()}'",
                    "confidence": float(best.confidence),
                    "trace": trace + [e.message for e in self.engine.meta.recent_trace()],
                }
            else:
                return {
                    "response": "No supporting evidence found in rules or known facts.",
                    "explanation": "No rule or fact supports the negated claim.",
                    "confidence": 0.0,
                    "trace": [e.message for e in self.engine.meta.recent_trace()],
                }

        # Try backward reasoning first
        facts, trace, conf = self.engine.reason_backward(goal)
        if facts:
            best = max(facts, key=lambda f: f.confidence)
            return {
                "response": self._evidence_text(best),
                "explanation": f"Proved: '{best.to_text()}' via backward reasoning.",
                "confidence": float(best.confidence),
                "trace": trace + [e.message for e in self.engine.meta.recent_trace()],
            }

        # Semantic memory fallback
        fact = self.engine.semantic.search_fact(query_embedding, threshold=0.6)
        if fact:
            return {
                "response": self._evidence_text(fact),
                "explanation": f"Found in semantic memory: '{fact.to_text()}'",
                "confidence": float(fact.confidence),
                "trace": [e.message for e in self.engine.meta.recent_trace()],
            }

        # Sensory memory fallback
        best_idx = None
        best_score = 0.0
        for idx, emb in enumerate(self.engine.sensory.entry_embeddings):
            sim = util.cos_sim(query_embedding, emb).item()
            if sim > best_score:
                best_score = sim
                best_idx = idx

        if best_idx is not None and best_score >= 0.35:
            best_entry = self.engine.sensory.entries[best_idx]
            return {
                "response": best_entry[:120],
                "explanation": f"Sensory memory match: {best_entry[:200]}...",
                "confidence": float(best_score),
                "trace": [e.message for e in self.engine.meta.recent_trace()],
            }

        # Unknown
        return {
            "response": "No supporting evidence found in reasoning, semantic memory, or sensory memory.",
            "explanation": f"No information found for '{text}'",
            "confidence": 0.0,
            "trace": [e.message for e in self.engine.meta.recent_trace()],
        }


# ==========================
# Core Knowledge Initialization
# ==========================

def build_core_knowledge(engine: CognitiveEngine):
    """
    Load basic commonsense knowledge + rules.
    """
    core_facts = [
        Fact(subject="fire", predicate="is", obj="hot", confidence=0.95, polarity=1, source="core"),
        Fact(subject="stove", predicate="is", obj="hot", confidence=0.9, polarity=1, source="core"),
        Fact(subject="boiling_water", predicate="is", obj="hot", confidence=0.94, polarity=1, source="core"),
        Fact(subject="steam", predicate="is", obj="hot", confidence=0.88, polarity=1, source="core"),
        Fact(subject="heated_metal", predicate="is", obj="hot", confidence=0.9, polarity=1, source="core"),
        Fact(subject="campfire_coals", predicate="is", obj="hot", confidence=0.93, polarity=1, source="core"),
        Fact(subject="oven_rack", predicate="is", obj="hot", confidence=0.87, polarity=1, source="core"),
        Fact(subject="candle_flame", predicate="is", obj="hot", confidence=0.91, polarity=1, source="core"),
        Fact(subject="space_heater", predicate="is", obj="hot", confidence=0.86, polarity=1, source="core"),
        Fact(subject="lava", predicate="is", obj="hot", confidence=0.99, polarity=1, source="core"),
        Fact(subject="radiator", predicate="is", obj="hot", confidence=0.84, polarity=1, source="core"),
        Fact(subject="sunlit_asphalt", predicate="is", obj="hot", confidence=0.78, polarity=1, source="core"),
        Fact(subject="ice", predicate="is", obj="cold", confidence=0.98, polarity=1, source="core"),
        Fact(subject="snow", predicate="is", obj="cold", confidence=0.95, polarity=1, source="core"),
        Fact(subject="freezer_pack", predicate="is", obj="cold", confidence=0.91, polarity=1, source="core"),
        Fact(subject="cold_water", predicate="is", obj="cold", confidence=0.86, polarity=1, source="core"),
        Fact(subject="frozen_metal", predicate="is", obj="cold", confidence=0.82, polarity=1, source="core"),
        Fact(subject="dry_ice", predicate="is", obj="cold", confidence=0.94, polarity=1, source="core"),
        Fact(subject="touching_fire", predicate="causes", obj="burn", confidence=0.95, polarity=1, source="core"),
        Fact(subject="touching_lava", predicate="causes", obj="burn", confidence=0.99, polarity=1, source="core"),
        Fact(subject="touching_steam", predicate="causes", obj="burn", confidence=0.9, polarity=1, source="core"),
        Fact(subject="touching_hot_pan", predicate="causes", obj="burn", confidence=0.93, polarity=1, source="core"),
        Fact(subject="touching_candle_flame", predicate="causes", obj="burn", confidence=0.92, polarity=1, source="core"),
        Fact(subject="touching_oven_rack", predicate="causes", obj="burn", confidence=0.88, polarity=1, source="core"),
        Fact(subject="touching_dry_ice", predicate="causes", obj="burn", confidence=0.78, polarity=1, source="core"),
        Fact(subject="knife", predicate="is", obj="sharp", confidence=0.96, polarity=1, source="core"),
        Fact(subject="broken_glass", predicate="is", obj="sharp", confidence=0.94, polarity=1, source="core"),
        Fact(subject="cactus_spine", predicate="is", obj="sharp", confidence=0.83, polarity=1, source="core"),
        Fact(subject="needle", predicate="is", obj="sharp", confidence=0.95, polarity=1, source="core"),
        Fact(subject="touching_knife_edge", predicate="causes", obj="cut", confidence=0.92, polarity=1, source="core"),
        Fact(subject="touching_broken_glass", predicate="causes", obj="cut", confidence=0.91, polarity=1, source="core"),
        Fact(subject="touching_needle_tip", predicate="causes", obj="cut", confidence=0.85, polarity=1, source="core"),
        Fact(subject="electric_socket", predicate="is", obj="dangerous", confidence=0.9, polarity=1, source="core"),
        Fact(subject="wet_floor", predicate="is", obj="slippery", confidence=0.92, polarity=1, source="core"),
        Fact(subject="ice_patch", predicate="is", obj="slippery", confidence=0.95, polarity=1, source="core"),
        Fact(subject="oil_spill", predicate="is", obj="slippery", confidence=0.94, polarity=1, source="core"),
        Fact(subject="running", predicate="on", obj="wet_floor", confidence=0.78, polarity=1, source="core"),
        Fact(subject="running", predicate="on", obj="ice_patch", confidence=0.8, polarity=1, source="core"),
        Fact(subject="touching_electric_socket", predicate="causes", obj="shock", confidence=0.93, polarity=1, source="core"),
        Fact(subject="shock", predicate="is", obj="dangerous", confidence=0.95, polarity=1, source="core"),
        Fact(subject="smoke", predicate="indicates", obj="fire", confidence=0.82, polarity=1, source="core"),
        Fact(subject="dark_clouds", predicate="indicate", obj="rain", confidence=0.74, polarity=1, source="core"),
        Fact(subject="rain", predicate="makes", obj="ground_wet", confidence=0.87, polarity=1, source="core"),
        Fact(subject="ground_wet", predicate="increases", obj="slip_risk", confidence=0.84, polarity=1, source="core"),
        Fact(subject="seatbelt", predicate="reduces", obj="injury_risk", confidence=0.89, polarity=1, source="core"),
        Fact(subject="helmet", predicate="reduces", obj="head_injury_risk", confidence=0.9, polarity=1, source="core"),
        Fact(subject="washing_hands", predicate="reduces", obj="germ_spread", confidence=0.88, polarity=1, source="core"),
        Fact(subject="soap", predicate="helps", obj="clean_hands", confidence=0.9, polarity=1, source="core"),
        Fact(subject="clean_hands", predicate="reduces", obj="infection_risk", confidence=0.86, polarity=1, source="core"),
        Fact(subject="sleep", predicate="improves", obj="focus", confidence=0.83, polarity=1, source="core"),
        Fact(subject="exercise", predicate="improves", obj="health", confidence=0.88, polarity=1, source="core"),
        Fact(subject="dehydration", predicate="causes", obj="fatigue", confidence=0.81, polarity=1, source="core"),
        Fact(subject="water", predicate="reduces", obj="dehydration", confidence=0.92, polarity=1, source="core"),
        Fact(subject="fatigue", predicate="reduces", obj="attention", confidence=0.85, polarity=1, source="core"),
        Fact(subject="low_attention", predicate="increases", obj="mistake_risk", confidence=0.79, polarity=1, source="core"),
        Fact(subject="traffic_light_red", predicate="means", obj="stop", confidence=0.97, polarity=1, source="core"),
        Fact(subject="traffic_light_green", predicate="means", obj="go", confidence=0.97, polarity=1, source="core"),
        Fact(subject="stop_sign", predicate="means", obj="stop", confidence=0.98, polarity=1, source="core"),
        Fact(subject="heavy_rain", predicate="reduces", obj="visibility", confidence=0.84, polarity=1, source="core"),
        Fact(subject="low_visibility", predicate="increases", obj="accident_risk", confidence=0.86, polarity=1, source="core"),
        Fact(subject="high_speed", predicate="increases", obj="accident_risk", confidence=0.9, polarity=1, source="core"),
    ]

    for fact in core_facts:
        engine.add_fact(fact)

    # Manual rules
    rule1 = Rule(
        name="hot_things_burn_rule",
        conditions=[RulePattern(subject="?x", predicate="is", obj="hot", polarity=1)],
        conclusion=RulePattern(subject="touching_?x", predicate="causes", obj="burn", polarity=1),
        confidence=0.9,
        source="manual",
    )

    rule2 = Rule(
        name="burn_implies_danger_rule",
        conditions=[RulePattern(subject="touching_?x", predicate="causes", obj="burn", polarity=1)],
        conclusion=RulePattern(subject="?x", predicate="is", obj="dangerous", polarity=1),
        confidence=0.9,
        source="manual",
    )

    # Negation rule example
    rule3 = Rule(
        name="cold_things_do_not_burn_rule",
        conditions=[RulePattern(subject="?x", predicate="is", obj="cold", polarity=1)],
        conclusion=RulePattern(subject="touching_?x", predicate="causes", obj="burn", polarity=-1),
        confidence=0.9,
        source="manual",
    )

    engine.add_rule(rule1)
    engine.add_rule(rule2)
    engine.add_rule(rule3)


# ==========================
# Main REPL
# ==========================

def main():
    engine = CognitiveEngine()
    api = AURAInterface(engine)

    # Load core knowledge
    build_core_knowledge(engine)
    engine.reason_forward_until_fixpoint(max_iterations=5)

    # Load Omniscience dataset
    ingest_omniscience(engine, max_items=50)

    print("\nAURA v7 ready. Type assertions or questions.")
    print("Examples:")
    print("  - 'fire is hot'")
    print("  - 'ice is cold'")
    print("  - 'does touching fire cause burn?'")
    print("  - 'does touching ice cause burn?'")
    print("  - 'is stove dangerous?'")
    print("  - 'do hot things NOT burn with fire?'")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ")
        except EOFError:
            break

        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        # Assertions
        if query.lower().startswith("assert "):
            fact = api.assert_fact(query[len("assert "):])
            engine.reason_forward_until_fixpoint(max_iterations=3)
            print(f"AURA: Asserted '{fact.to_text()}' (conf={fact.confidence:.2f})\n")
            continue

        # Queries
        response = api.query(query)
        print(f"AURA Response: {response['response']}")
        print(f"Explanation: {response['explanation']}")
        print(f"Confidence: {response['confidence']:.2f}")

        if response.get("trace"):
            print("Recent reasoning trace:")
            for line in response["trace"][-10:]:
                print("  -", line)
        print()


if __name__ == "__main__":
    main()
