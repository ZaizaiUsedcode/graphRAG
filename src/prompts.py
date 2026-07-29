"""Prompts for GraphRAG extraction and search."""

EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and the
supported entity types below, identify all supported entities from the text and
all relationships among the identified entities.

Supported entity types:
Organization, Person, Policy, Process, System, Concept, Document, Role,
Regulation, Event, Location, Product, Technology

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity in English and capitalized. Use the entity's
  established official English name when available; otherwise provide a
  consistent English translation or transliteration
- entity_type: One of the supported entity types listed above
- entity_description: Comprehensive description of the entity's attributes,
  activities, and significance, written in English

Format each entity as
("entity"<|><entity_name><|><entity_type><|><entity_description>).

2. From the entities identified in step 1, identify all pairs of
(source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: Name of the source entity exactly as identified in step 1
- target_entity: Name of the target entity exactly as identified in step 1
- relationship_description: Explanation of why the entities are related,
  explicitly stating the subject, action, object, and applicable conditions,
  written in English
- relationship_type: A specific, action-oriented verb phrase describing the
  relationship (for example, "regulates", "implements", "depends_on",
  "manages", "contains", "requires", "produces", or "authorizes"). Write it
  in English using lowercase snake_case
- relationship_strength: An integer from 1 to 10 indicating the strength or
  importance of the relationship

Avoid generic relationship types such as "related", "associated", "connected",
"mentions", or their equivalents in the input language. Use a specific English
verb phrase that describes the actual relationship.

Format each relationship as
("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_type><|><relationship_strength>).

3. Return all entity names, entity descriptions, relationship descriptions, and
relationship types in English, regardless of the input language. Return a
single list of all entities and relationships identified in steps 1 and 2. Use
**##** as the list delimiter. Do not include commentary outside the formatted
records.

4. When finished, output <|COMPLETE|>

######################
-Examples-
######################
Example 1:

Text:
The Anti-Money Laundering Department is responsible for monitoring suspicious transactions. All transactions over $10,000 must be reported to the Financial Intelligence Unit within 24 hours. The department uses the AML Detection System to flag potential violations of the Bank Secrecy Act.

################
Output:
("entity"<|>Anti-Money Laundering Department<|>Organization<|>Department responsible for monitoring and investigating suspicious financial transactions)
##
("entity"<|>Financial Intelligence Unit<|>Organization<|>Regulatory body that receives reports of suspicious financial activities)
##
("entity"<|>AML Detection System<|>System<|>Automated system used to detect and flag potential money laundering activities)
##
("entity"<|>Bank Secrecy Act<|>Regulation<|>Federal law requiring financial institutions to assist government agencies in detecting money laundering)
##
("entity"<|>Suspicious Transaction Reporting<|>Process<|>Mandatory reporting process for transactions exceeding $10,000 threshold)
##
("relationship"<|>Anti-Money Laundering Department<|>Financial Intelligence Unit<|>The AML Department must report suspicious transactions to the Financial Intelligence Unit within 24 hours<|>reports_to<|>8)
##
("relationship"<|>Anti-Money Laundering Department<|>AML Detection System<|>The department uses the AML Detection System as its primary tool for identifying suspicious activities<|>operates<|>9)
##
("relationship"<|>AML Detection System<|>Bank Secrecy Act<|>The system is designed to detect violations of the Bank Secrecy Act requirements<|>enforces<|>7)
##
("relationship"<|>Anti-Money Laundering Department<|>Suspicious Transaction Reporting<|>The department is responsible for executing the suspicious transaction reporting process<|>executes<|>8)
<|COMPLETE|>

######################
Example 2:

Text:
The Chief Compliance Officer oversees the implementation of Know Your Customer procedures. New customer accounts must go through identity verification before approval. The KYC Policy requires collecting government-issued ID and proof of address from all customers.

################
Output:
("entity"<|>Chief Compliance Officer<|>Role<|>Senior executive responsible for ensuring organizational compliance with regulations and policies)
##
("entity"<|>Know Your Customer<|>Process<|>Regulatory process for verifying customer identity and assessing potential risks)
##
("entity"<|>KYC Policy<|>Policy<|>Internal policy document specifying requirements for customer identification and verification)
##
("entity"<|>Identity Verification<|>Process<|>Process of confirming customer identity through document verification)
##
("entity"<|>Customer Account<|>Concept<|>Financial account opened by customers requiring identity verification)
##
("relationship"<|>Chief Compliance Officer<|>Know Your Customer<|>The CCO is responsible for overseeing the implementation and execution of KYC procedures<|>oversees<|>9)
##
("relationship"<|>Customer Account<|>Identity Verification<|>New customer accounts must complete identity verification before being approved<|>requires<|>10)
##
("relationship"<|>KYC Policy<|>Identity Verification<|>The KYC Policy defines the requirements and standards for identity verification<|>specifies<|>8)
##
("relationship"<|>Know Your Customer<|>KYC Policy<|>The KYC process is governed and defined by the KYC Policy<|>governed_by<|>9)
<|COMPLETE|>

######################
-Real Data-
######################
Entity_types:
Organization, Person, Policy, Process, System, Concept, Document, Role,
Regulation, Event, Location, Product, Technology

Text:
{chunk}
################
Output:
"""

CONTINUE_PROMPT = (
    "MANY entities and relationships were missed in the last extraction. "
    "Remember to ONLY emit entities that match one of the previously specified "
    "types. Add them below using the same format:\n"
)

LOOP_PROMPT = (
    "It appears some entities and relationships may have still been missed. "
    "Answer Y if there are still entities or relationships that need to be "
    "added, or N if there are none. Please answer with a single letter Y or N.\n"
)

ANSWER_PROMPT = """Based on the following document content and entity relationships, answer the user's question.
If the information is insufficient, please state that clearly. Do not fabricate information.
Always write the answer in English, regardless of the language of the question or source material.

[Relevant Document Chunks]
{context_chunks}

[Entity Relationships (Cross-document Logic Chains)]
{context_relations}

[User Question]
{query}

Please provide an accurate, well-supported answer:"""

COMMUNITY_SUMMARY_PROMPT = """You are a professional knowledge graph analyst. Please generate a concise summary for the following community based on its entities and relationships.
Always write the summary in English, regardless of the language of the source material.

Community ID: {community_id}
Level: {level}

[Entities in this Community]
{entities}

[Relationships within this Community]
{relationships}

Please generate a concise summary (100-200 words) that captures:
1. The main theme or domain of this community
2. Core concepts and their significance
3. Key relationships and dependencies

Community Summary:"""

GLOBAL_MAP_PROMPT = """You are a professional knowledge graph analyst. Based on the following community information, evaluate its relevance to the user's question and generate a partial answer based on the community content.
Write the partial answer in English, regardless of the language of the question or source material.

[User Question]
{query}

[Community Summary]
{community_summary}

[Entities in this Community]
{entities}

[Relationships within this Community]
{relationships}

Please output in the following JSON format:
{{
  "relevance": <integer from 1-5, where 5 means highly relevant and 1 means almost irrelevant>,
  "answer": "<partial answer to the question based on this community's information, or 'No relevant information' if not applicable>"
}}"""

GLOBAL_REDUCE_PROMPT = """You are a professional knowledge graph analyst. Please synthesize the following partial answers from multiple communities into a complete, coherent final answer.
Always write the final answer in English, regardless of the language of the question or source material.

[User Question]
{query}

[Partial Answers from Communities]
{partial_answers}

Please synthesize a complete and accurate answer with the following requirements:
1. Integrate information from different communities, avoiding redundancy
2. Maintain logical coherence
3. If information is insufficient, state that clearly
4. Do not fabricate information without supporting evidence

Final Answer:"""

QUERY_ROUTER_PROMPT = """You are a query routing assistant. Analyze the user's question and determine which search mode should be used.

[User Question]
{query}

Search Mode Descriptions:
- local: Best for specific, detailed questions such as "What is X?", "How to do Y?", "What is the relationship between X and Y?"
- global: Best for broad, comprehensive questions such as "What is the overall system architecture?", "What are the main topics covered?", "Summarize..."

Please output only one word: local or global"""
