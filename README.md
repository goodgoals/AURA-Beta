# AURA (Beta) — Architecture for Unified Reasoning & Awareness
AURA is an experimental, non-LLM cognitive architecture focused on explicit, verifiable reasoning rather than probabilistic text generation. The system is designed as a modular reasoning engine composed of distinct memory layers, rule systems, and inference pipelines, with an emphasis on correctness, transparency, and epistemic humility. AURA does not attempt to predict the “most likely” answer to a query. Instead, it evaluates whether a conclusion can be justified based on known facts, rules, and external evidence, and will explicitly return uncertainty when justification is insufficient.
At its core, AURA treats intelligence as a system behavior, not a monolithic model. Knowledge is represented through structured facts, relationships, and confidence values rather than latent token statistics. Reasoning occurs through deterministic or semi-deterministic processes such as rule application, dependency tracing, contradiction detection, and evidence aggregation. Each conclusion can be traced back to its originating data sources, rules, or assumptions, allowing developers to audit, debug, and refine reasoning pathways over time.
AURA’s architecture separates cognition into multiple layers, including raw data ingestion (sensory memory), transient hypothesis and reasoning chains (working memory), persistent structured knowledge (long-term memory), and meta-cognitive tracking (e.g., confidence decay, contradictions, unresolved queries). This layered approach allows the system to reason incrementally, revise beliefs when new evidence is introduced, and avoid over-generalization. Unlike LLM-based systems, AURA does not collapse all reasoning into vector similarity; embeddings, when used, are strictly auxiliary and never authoritative.
The system is designed to integrate large external knowledge sources such as Wikimedia/Wikipedia datasets, but ingestion is treated as evidence acquisition, not truth acceptance. Incoming data is parsed, normalized, and stored as candidate knowledge that must be validated or contextualized through rules and cross-reference mechanisms. AURA explicitly avoids answering questions purely because related text exists; relevance alone is not sufficient for verification.
AURA is currently in Beta and should be considered unstable and under active development. APIs, internal representations, and reasoning pipelines are subject to change as the architecture evolves. Performance, scalability, and completeness are not yet guaranteed, and incorrect or incomplete conclusions are possible. The project is intended for developers and researchers interested in exploring alternative approaches to AI reasoning beyond transformer-based language models, particularly those focused on symbolic reasoning, hybrid systems, and explainable AI.
The long-term goal of AURA is not to compete with LLMs in fluency, but to explore a different axis of intelligence—one centered on integrity, traceability, and genuine reasoning. Contributions, experiments, and architectural critiques are encouraged, especially from developers interested in cognitive systems, knowledge representation, and non-statistical approaches to artificial intelligence.
# How to test:
Keep in mind that AURA is still in beta and a stable release isn't yet to be published. Use at your own caution and don't expect everything to be perfect yet.
In order to test AURA, first get your test device ready. Install Python 3.12 or above on your machine to run the code if not already. Then, clone the repository using git clone. Then, open up a terminal or command prompt window and type in: cd AURA-Beta
Then, type: cd src-code
Lastly, type in: python AURA-main-code.py
Then, AURA Beta will run on your device so you can test it.

# Testing instructions for cloud compute
If you are using cloud compute like Google Colab, Kaggle, Hugging Face Spaces, etc, follow the instructions given below for your cloud compute provider:

# Google Colab and Kaggle Notebooks
First, make a blank notebook in Google Colab or Kaggle. Then paste the contents of AURA-Beta/src-code/AURA-main-code.py onto your code cell. (Contents can be viewed in GitHub). Now, run the code cell and easily test AURA Beta without installing or running anything on your local machine. This is guaranteed to work on Google Colab using T4 GPU because it was one of the testing environments we used to test AURA Beta.

# Hugging Face Spaces
If you are using Hugging Face Spaces, you have a repo like setup (similar to GitHub) which makes your job quite easy. All you have to do is create a Streamlit Template Space (Why Streamlit when there is going to be no Streamlit UI?: The Streamlit Template Space comes with a prebuilt Dockerfile that is perfect for our purposes). So make a Streamlit Template Space with the configuration of your choice. Now, go to your repository and follow these instructions: 

Delete the existing app.py
Do not do anything to the Dockerfile: Leave it as is: It's already preconfigured the way we need it
Now come over to the AURA-Beta GitHub Repository and copy the contents of src-code/AURA-main-code.py.
Now create a new file in your Hugging Face Repository called "app.py" and paste the previously copied contents as the contents of your app.py
Now, after you make the file, Hugging Face with automatically start building your space. Don't mind it. It will error out because you need to do an extra step
Go to the "requirements.txt" file and edit it: Remove all existing items and copy this list: 


torch
sentence-transformers
networkx
datasets
numpy
scikit-learn
tqdm
transformers


Then, paste this list as the contents of "requirements.txt"
Now save it and Hugging Face will start building
It might take a while, but then, it will say Starting and start your space
Then you can test AURA-Beta as much as you want!

Hugging Face might be the most tedious, but also the most rewarding, because now, you have a website program just to try AURA Beta out whenever you want to!

