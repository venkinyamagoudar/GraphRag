import os
from dotenv import load_dotenv
load_dotenv()

# Now import everything else
import pandas as pd
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import pickle
import asyncio
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
from datetime import datetime
import re
import tiktoken
from asyncio import Semaphore

# LangChain imports
from langchain_openai import AzureChatOpenAI
from langchain_mistralai import MistralAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import BaseOutputParser
from langchain.prompts import PromptTemplate


# Visualization
from pyvis.network import Network

# Community detection - Leiden algorithm (preferred)
try:
    import leidenalg
    import igraph as ig
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False
    leidenalg = None
    ig = None

# Fallback community detection algorithms
try:
    import community.community_louvain as community_louvain
    HAS_PYTHON_LOUVAIN = True
except ImportError:
    try:
        import community as community_louvain
        HAS_PYTHON_LOUVAIN = True
    except ImportError:
        HAS_PYTHON_LOUVAIN = False
        community_louvain = None

# NetworkX built-in community detection algorithms (final fallback)
from networkx.algorithms.community import louvain_communities, greedy_modularity_communities



# Disable verbose HTTP logging to reduce noise and save costs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    name: str
    type: str
    description: str
    source_id: str

@dataclass
class Relationship:
    source: str
    target: str
    description: str
    strength: float
    source_id: str

@dataclass
class Claim:
    subject: str
    predicate: str
    object: str
    description: str
    source_id: str

@dataclass
class TextUnit:
    id: str
    text: str
    n_tokens: int
    document_ids: List[str]
    entity_ids: List[str]
    relationship_ids: List[str]

@dataclass
class CommunityReport:
    community_id: str
    level: int
    title: str
    summary: str
    rating: float
    rating_explanation: str
    findings: List[Dict[str, str]]
    full_content: str
    rank: float

class RelationshipOutputParser(BaseOutputParser):
    """Custom parser for Microsoft GraphRAG relationship format"""
    
    def parse(self, text: str) -> Tuple[List[Entity], List[Relationship]]:
        """Parse entities and relationships from text"""
        return self._parse_extraction_response(text, "unknown")
    
    def get_format_instructions(self) -> str:
        """Return format instructions for the parser"""
        return """
        Format entities as: ("entity"<|><entity name><|><entity type><|><entity description>)
        Format relationships as: ("relationship"<|><source entity><|><target entity><|><relationship description><|><relationship strength>)
        Use <|RECORD|> as delimiter between records.
        End with <|COMPLETE|>
        """
    
    def _parse_extraction_response(self, response: str, source_id: str) -> Tuple[List[Entity], List[Relationship]]:
        """Your existing parsing logic"""
        entities = []
        relationships = []
        
        records = response.split("<|RECORD|>")
        
        for record in records:
            record = record.strip()
            if not record or "<|COMPLETE|>" in record:
                continue
                
            if record.startswith('("entity"'):
                try:
                    content = record[record.find('(')+1:record.rfind(')')]
                    parts = content.split('<|>')
                    
                    if len(parts) >= 4:
                        entity = Entity(
                            name=parts[1].strip().upper(),
                            type=parts[2].strip(),
                            description=parts[3].strip(),
                            source_id=source_id
                        )
                        entities.append(entity)
                except Exception:
                    continue
                    
            elif record.startswith('("relationship"'):
                try:
                    content = record[record.find('(')+1:record.rfind(')')]
                    parts = content.split('<|>')
                    
                    if len(parts) >= 5:
                        relationship = Relationship(
                            source=parts[1].strip().upper(),
                            target=parts[2].strip().upper(),
                            description=parts[3].strip(),
                            strength=float(parts[4].strip()),
                            source_id=source_id
                        )
                        relationships.append(relationship)
                except Exception:
                    continue
        
        return entities, relationships

class OptimizedGraphRAG:
    """Cost-optimized Microsoft GraphRAG with batch processing and minimal API calls"""
    
    def __init__(self, 
                 azure_endpoint: str,
                 azure_key: str,
                 azure_deployment: str,
                 mistral_key: str,
                 api_version: str = "2024-02-15-preview",
                 max_concurrent_requests: int = 5,
                 request_delay: float = 0.1):
        
        self.llm = AzureChatOpenAI(
            deployment_name=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            openai_api_key=azure_key,
            temperature=0,
            max_tokens=4000,
            request_timeout=60
        )
        
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=mistral_key
        )
        
        # Rate limiting configuration
        self.semaphore = Semaphore(max_concurrent_requests)
        self.request_delay = request_delay
        
        # Initialize storage
        self.text_units: List[TextUnit] = []
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.claims: Dict[str, Claim] = {}
        self.graph = nx.Graph()
        self.community_hierarchy: Dict[int, Dict[str, List[str]]] = {}
        self.community_reports: Dict[str, CommunityReport] = {}
        
        # Token counting and cost tracking
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.total_tokens_used = 0
        self.api_calls_made = 0
        
        # ADD THIS: Initialize the fixing parser
        self.relationship_parser = RelationshipOutputParser()
        self.fixing_parser = OutputFixingParser.from_llm(
            llm=self.llm,
            parser=self.relationship_parser,
            max_retries=2
        )
        
        logger.info("✅ Optimized GraphRAG initialized with rate limiting")
        
        logger.info("✅ Optimized GraphRAG initialized with rate limiting")

    # ==================== MICROSOFT'S EXACT PROMPTS ====================
    
    def get_entity_extraction_prompt(self) -> PromptTemplate:
        """Microsoft's exact entity extraction prompt from the paper"""
        return PromptTemplate(
            template="""---Goal---
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity name: Name of the entity, capitalized
- entity type: One of the following types: [PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT, TECHNOLOGY, PRODUCT, SERVICE, OTHER]
- entity description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity name><|><entity type><|><entity description>)

2. From the entities identified in step 1, identify all pairs of (source entity, target entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source entity: name of the source entity, as identified in step 1
- target entity: name of the target entity, as identified in step 1
- relationship description: explanation as to why you think the source entity and the target entity are related to each other
- relationship strength: a numeric score indicating strength of the relationship between the source entity and target entity
Format each relationship as ("relationship"<|><source entity><|><target entity><|><relationship description><|><relationship strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2.
Use **<|RECORD|>** as the list delimiter.

4. When finished, output <|COMPLETE|>

---Real Data---
Entity types: PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT, TECHNOLOGY, PRODUCT, SERVICE, OTHER
Input: {input_text}
Output:""",
            input_variables=["input_text"]
        )

    def get_claim_extraction_prompt(self) -> PromptTemplate:
        """Microsoft's claim extraction prompt"""
        return PromptTemplate(
            template="""---Goal---
Given a text document and a list of entities, extract all factual claims about these entities.

---Steps---
1. For each entity mentioned in the text, identify all factual claims made about that entity.
2. A claim should be a verifiable statement of fact, not an opinion or speculation.
3. Format each claim as ("claim"<|><subject><|><predicate><|><object><|><description>)

---Real Data---
Input: {input_text}
Entities: {entities}
Output:""",
            input_variables=["input_text", "entities"]
        )

    def get_community_summary_prompt(self) -> PromptTemplate:
        """Microsoft's community summarization prompt"""
        return PromptTemplate(
            template="""---Role---
You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

---Goal---
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact.

---Report Structure---
The report should include the following sections:
- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below.

Return output as a well-formed JSON-formatted string with the following format:
{{
    "title": <report title>,
    "summary": <executive summary>,
    "rating": <impact severity rating>,
    "rating_explanation": <rating explanation>,
    "findings": [
        {{
            "summary": <insight 1 summary>,
            "explanation": <insight 1 explanation>
        }},
        {{
            "summary": <insight 2 summary>,
            "explanation": <insight 2 explanation>
        }}
    ]
}}

---Real Data---
Entities:
{entities}

Relationships:
{relationships}

Claims:
{claims}

Output:""",
            input_variables=["entities", "relationships", "claims"]
        )

    def get_global_search_prompt(self) -> PromptTemplate:
        """Microsoft's global search prompt"""
        return PromptTemplate(
            template="""---Role---
You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.

---Goal---
Generate a response that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset, and incorporate any relevant general knowledge.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

---Analyst Reports---
{reports}

---Question---
{question}

Output:""",
            input_variables=["reports", "question"]
        )

    def get_local_search_prompt(self) -> PromptTemplate:
        """Microsoft's local search prompt"""
        return PromptTemplate(
            template="""---Role---
You are a helpful assistant responding to questions about data in the tables provided.

---Goal---
Generate a response that responds to the user's question, summarize all relevant information in the input data tables appropriate for the response, and incorporate any relevant general knowledge.

---Data tables---
{context_data}

---Question---
{question}

Output:""",
            input_variables=["context_data", "question"]
        )

    # ==================== COST-OPTIMIZED API CALLS ====================
    
    async def make_llm_call_with_rate_limiting(self, prompt_chain, input_data: dict) -> str:
        """Make LLM call with rate limiting and cost tracking"""
        async with self.semaphore:
            try:
                # Add delay to avoid rate limiting
                await asyncio.sleep(self.request_delay)
                
                # Count tokens before API call
                input_text = str(input_data)
                input_tokens = len(self.encoding.encode(input_text))
                
                # Make API call
                response = await prompt_chain.ainvoke(input_data)
                
                # Count output tokens and track costs
                output_tokens = len(self.encoding.encode(response.content))
                self.total_tokens_used += input_tokens + output_tokens
                self.api_calls_made += 1
                
                if self.api_calls_made % 50 == 0:
                    logger.info(f"💰 Cost tracking: {self.api_calls_made} API calls, {self.total_tokens_used:,} tokens used")
                
                return response.content
                
            except Exception as e:
                logger.error(f"❌ API call failed: {e}")
                await asyncio.sleep(2)  # Backoff on error
                raise

    # ==================== TEXT PROCESSING ====================
    
    def chunk_documents(self, documents: List[str], chunk_size: int = 600, chunk_overlap: int = 100) -> List[TextUnit]:
        """Chunk documents into text units following Microsoft's approach"""
        logger.info(f"🔪 Chunking documents into {chunk_size}-token units")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: len(self.encoding.encode(x)),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        text_units = []
        for doc_id, document in enumerate(documents):
            chunks = text_splitter.split_text(document)
            
            for chunk_id, chunk in enumerate(chunks):
                unit_id = f"unit_{doc_id}_{chunk_id}"
                n_tokens = len(self.encoding.encode(chunk))
                
                text_unit = TextUnit(
                    id=unit_id,
                    text=chunk,
                    n_tokens=n_tokens,
                    document_ids=[str(doc_id)],
                    entity_ids=[],
                    relationship_ids=[]
                )
                text_units.append(text_unit)
        
        logger.info(f"✅ Created {len(text_units)} text units")
        return text_units

    # ==================== BATCH PROCESSING WITH RATE LIMITING ====================
    
    async def extract_entities_and_relationships_batch(self, text_unit: TextUnit) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships with auto-fixing"""
        entity_prompt = self.get_entity_extraction_prompt()
        entity_chain = entity_prompt | self.llm
        
        try:
            response = await self.make_llm_call_with_rate_limiting(
                entity_chain, 
                {"input_text": text_unit.text}
            )
            
            # Use the fixing parser instead of direct parsing
            try:
                entities, relationships = self.fixing_parser.parse(response) 
                # Debug zero extractions
                if len(entities) == 0 and len(relationships) == 0:
                    logger.debug(f"Zero extraction from chunk: {text_unit.text[:100]}...")
                
                logger.info(f"✅ Successfully parsed {len(entities)} entities, {len(relationships)} relationships")    
                return entities, relationships
            
            except Exception as e:
                logger.warning(f"⚠️ Fixing parser failed, using fallback: {e}")
                # Fallback to your robust manual parsing
                return self._parse_extraction_response_robust(response, text_unit.id)
                
        except Exception as e:
            logger.error(f"❌ Error extracting from unit {text_unit.id}: {e}")
            return [], []

    async def extract_claims_batch(self, text_unit: TextUnit, entities: List[Entity]) -> List[Claim]:
        """Extract claims with rate limiting"""
        if not entities:
            return []
            
        entity_names = [e.name for e in entities]
        claim_prompt = self.get_claim_extraction_prompt()
        claim_chain = claim_prompt | self.llm
        
        try:
            response = await self.make_llm_call_with_rate_limiting(
                claim_chain,
                {
                    "input_text": text_unit.text,
                    "entities": ", ".join(entity_names)
                }
            )
            return self._parse_claims_response(response, text_unit.id)
        except Exception as e:
            logger.error(f"❌ Error extracting claims from unit {text_unit.id}: {e}")
            return []

    async def process_text_units_in_batches(self, text_units: List[TextUnit], batch_size: int = 10) -> Tuple[List[Entity], List[Relationship], List[Claim]]:
        """Process text units in batches with rate limiting and checkpointing"""
        logger.info(f"🔄 Processing {len(text_units)} text units in batches of {batch_size}")
        
        all_entities = []
        all_relationships = []
        all_claims = []
        
        # Process in batches
        for i in range(0, len(text_units), batch_size):
            batch = text_units[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(text_units) - 1) // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} units)")
            
            # Process batch with controlled concurrency
            batch_entities = []
            batch_relationships = []
            batch_claims = []
            
            # Create tasks for entity/relationship extraction
            entity_tasks = [self.extract_entities_and_relationships_batch(unit) for unit in batch]
            
            # Execute with rate limiting
            batch_results = await asyncio.gather(*entity_tasks, return_exceptions=True)
            
            # Process results and extract claims
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Error processing unit {batch[j].id}: {result}")
                    continue
                    
                entities, relationships = result
                batch_entities.extend(entities)
                batch_relationships.extend(relationships)
                
                # Extract claims for entities found
                if entities:
                    claims = await self.extract_claims_batch(batch[j], entities)
                    batch_claims.extend(claims)
            
            # Add batch results to totals
            all_entities.extend(batch_entities)
            all_relationships.extend(batch_relationships)
            all_claims.extend(batch_claims)
            
            # Save checkpoint every 10 batches
            if batch_num % 10 == 0:
                await self.save_checkpoint(batch_num, all_entities, all_relationships, all_claims)
                logger.info(f"💾 Checkpoint saved at batch {batch_num}")
        
        logger.info(f"✅ Batch processing complete: {len(all_entities)} entities, {len(all_relationships)} relationships, {len(all_claims)} claims")
        return all_entities, all_relationships, all_claims

    # ==================== CHECKPOINT SYSTEM ====================
    
    async def save_checkpoint(self, batch_num: int, entities: List[Entity], relationships: List[Relationship], claims: List[Claim]):
        """Save progress checkpoint"""
        checkpoint_data = {
            'batch_num': batch_num,
            'entities': [asdict(e) for e in entities],
            'relationships': [asdict(r) for r in relationships], 
            'claims': [asdict(c) for c in claims],
            'total_tokens_used': self.total_tokens_used,
            'api_calls_made': self.api_calls_made,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = f"checkpoint_batch_{batch_num}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)

    async def load_checkpoint(self, checkpoint_file: str) -> Tuple[List[Entity], List[Relationship], List[Claim], int]:
        """Load from checkpoint"""
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        
        entities = [Entity(**e) for e in data['entities']]
        relationships = [Relationship(**r) for r in data['relationships']]
        claims = [Claim(**c) for c in data['claims']]
        
        self.total_tokens_used = data.get('total_tokens_used', 0)
        self.api_calls_made = data.get('api_calls_made', 0)
        
        return entities, relationships, claims, data['batch_num']

    # ==================== PARSING METHODS ====================
    
    def _parse_extraction_response_robust(self, response: str, source_id: str) -> Tuple[List[Entity], List[Relationship]]:
        """Robust fallback parsing with better error handling"""
        entities = []
        relationships = []
        
        # More flexible parsing that handles truncated responses
        lines = response.split('\n')
        current_record = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Handle multi-line records
            if line.startswith('("entity"') or line.startswith('("relationship"'):
                if current_record:
                    # Process previous record
                    self._process_record(current_record, entities, relationships, source_id)
                current_record = line
            elif current_record and not line.startswith('<|RECORD|>') and not line.startswith('<|COMPLETE|>'):
                # Continue building current record
                current_record += " " + line
            elif line.startswith('<|RECORD|>'):
                if current_record:
                    self._process_record(current_record, entities, relationships, source_id)
                    current_record = ""
        
        # Process final record
        if current_record:
            self._process_record(current_record, entities, relationships, source_id)
        
        return entities, relationships

    def _parse_claims_response(self, response: str, source_id: str) -> List[Claim]:
        """Parse claims from LLM response"""
        claims = []
        records = response.split("<|RECORD|>")
        
        for record in records:
            record = record.strip()
            if not record or "<|COMPLETE|>" in record:
                continue
                
            if record.startswith('("claim"'):
                try:
                    content = record[record.find('(')+1:record.rfind(')')]
                    parts = content.split('<|>')
                    
                    if len(parts) >= 5:
                        claim = Claim(
                            subject=parts[1].strip(),
                            predicate=parts[2].strip(),
                            object=parts[3].strip(),
                            description=parts[4].strip(),
                            source_id=source_id
                        )
                        claims.append(claim)
                except Exception as e:
                    logger.warning(f"Failed to parse claim: {record[:100]}...")
        
        return claims
    
    def _process_record(self, record: str, entities: List[Entity], relationships: List[Relationship], source_id: str):
        """Process a single record with robust error handling"""
        try:
            if record.startswith('("entity"'):
                # Extract entity even if incomplete
                parts = record.split('<|>')
                if len(parts) >= 3:
                    name = parts[1].strip().upper() if len(parts) > 1 else "UNKNOWN"
                    entity_type = parts[2].strip() if len(parts) > 2 else "OTHER"
                    description = parts[3].strip() if len(parts) > 3 else "No description"
                    
                    entity = Entity(name=name, type=entity_type, description=description, source_id=source_id)
                    entities.append(entity)
                    
            elif record.startswith('("relationship"'):
                # Extract relationship even if incomplete
                parts = record.split('<|>')
                if len(parts) >= 4:
                    source = parts[1].strip().upper() if len(parts) > 1 else "UNKNOWN"
                    target = parts[2].strip().upper() if len(parts) > 2 else "UNKNOWN"
                    description = parts[3].strip() if len(parts) > 3 else "Related"
                    
                    # Try to extract strength, default to 5.0 if missing
                    try:
                        strength = float(parts[4].strip()) if len(parts) > 4 else 5.0
                    except:
                        strength = 5.0
                    
                    relationship = Relationship(
                        source=source, target=target, description=description, 
                        strength=strength, source_id=source_id
                    )
                    relationships.append(relationship)
                    
        except Exception as e:
            logger.debug(f"Failed to process record: {record[:50]}... Error: {e}")

    # ==================== GRAPH CONSTRUCTION ====================
    
    def build_knowledge_graph(self):
        """Build the knowledge graph from extracted entities and relationships"""
        logger.info("🕸️ Building knowledge graph...")
        
        # Add entities as nodes
        for entity_id, entity in self.entities.items():
            self.graph.add_node(
                entity.name,
                type=entity.type,
                description=entity.description,
                entity_id=entity_id
            )
        
        # Add relationships as edges
        for rel_id, relationship in self.relationships.items():
            if relationship.source in self.graph and relationship.target in self.graph:
                self.graph.add_edge(
                    relationship.source,
                    relationship.target,
                    description=relationship.description,
                    strength=relationship.strength,
                    relationship_id=rel_id
                )
        
        logger.info(f"✅ Knowledge graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    # ==================== YOUR UPDATED COMMUNITY DETECTION ====================
    
    def detect_hierarchical_communities(self, max_levels: int = 4) -> Dict[int, Dict[str, List[str]]]:
        """Detect hierarchical communities using Leiden algorithm (preferred) with fallbacks"""
        logger.info("🔍 Detecting hierarchical communities...")
        if self.graph.number_of_nodes() == 0:
            return {}
        
        # Convert to undirected for community detection
        undirected_graph = self.graph.to_undirected()
        
        def _detect_communities_leiden(graph):
            """Use Leiden algorithm for community detection with correct partition type"""
            # Convert NetworkX graph to igraph
            edges = list(graph.edges())
            nodes = list(graph.nodes())
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}
            
            # Create igraph Graph
            g = ig.Graph()
            g.add_vertices(len(nodes))
            if edges:
                edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in edges]
                g.add_edges(edge_list)
            
            # Use RBConfigurationVertexPartition instead of ModularityVertexPartition
            partition = leidenalg.find_partition(
                g, 
                leidenalg.RBConfigurationVertexPartition,  # Changed from ModularityVertexPartition
                resolution_parameter=1.0
            )
            
            # Convert back to NetworkX node names
            communities = defaultdict(list)
            for idx, community_id in enumerate(partition.membership):
                communities[community_id].append(nodes[idx])
            
            return list(communities.values())

        
        def _detect_communities_fallback(graph, resolution=1.0):
            """Fallback community detection methods"""
            try:
                if HAS_PYTHON_LOUVAIN and community_louvain:
                    partition = community_louvain.best_partition(graph, resolution=resolution)
                    communities = defaultdict(list)
                    for node, comm_id in partition.items():
                        communities[comm_id].append(node)
                    return list(communities.values())
                else:
                    return list(louvain_communities(graph, resolution=resolution))
            except Exception:
                return list(greedy_modularity_communities(graph))
        
        # Level 0: Root communities
        try:
            if HAS_LEIDEN:
                logger.info("Using Leiden algorithm for community detection")
                communities_level_0 = _detect_communities_leiden(undirected_graph)
            else:
                logger.info("Leiden not available, using fallback algorithms")
                communities_level_0 = _detect_communities_fallback(undirected_graph)
            
            self.community_hierarchy[0] = {
                f"community_0_{i}": list(community)
                for i, community in enumerate(communities_level_0)
            }
            
        except Exception as e:
            logger.warning(f"Primary community detection failed: {e}, using greedy modularity")
            communities_level_0 = greedy_modularity_communities(undirected_graph)
            self.community_hierarchy[0] = {
                f"community_0_{i}": list(community)
                for i, community in enumerate(communities_level_0)
            }
        
        # Hierarchical subdivision for levels 1-3
        for level in range(1, max_levels):
            self.community_hierarchy[level] = {}
            for parent_comm_id, parent_nodes in self.community_hierarchy[level-1].items():
                if len(parent_nodes) <= 3:
                    self.community_hierarchy[level][parent_comm_id] = parent_nodes
                    continue
                
                # Create subgraph and detect sub-communities
                subgraph = undirected_graph.subgraph(parent_nodes)
                if subgraph.number_of_nodes() <= 1:
                    self.community_hierarchy[level][parent_comm_id] = parent_nodes
                    continue
                    
                try:
                    if HAS_LEIDEN:
                        # Use RBConfigurationVertexPartition for subdivision with higher resolution
                        edges = list(subgraph.edges())
                        nodes = list(subgraph.nodes())
                        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
                        
                        g = ig.Graph()
                        g.add_vertices(len(nodes))
                        if edges:
                            edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in edges]
                            g.add_edges(edge_list)
                        
                        partition = leidenalg.find_partition(
                            g, 
                            leidenalg.RBConfigurationVertexPartition,
                            resolution_parameter=1.2  # Higher resolution for subdivision
                        )
                        
                        communities = defaultdict(list)
                        for idx, community_id in enumerate(partition.membership):
                            communities[community_id].append(nodes[idx])
                        
                        sub_communities = list(communities.values())
                    else:
                        sub_communities = _detect_communities_fallback(subgraph, resolution=1.2)
                    
                    # Only create subcommunities if we found more than one
                    if len(sub_communities) > 1 and all(len(comm) > 0 for comm in sub_communities):
                        for j, sub_comm in enumerate(sub_communities):
                            if len(sub_comm) > 0:  # Skip empty communities
                                sub_comm_id = f"{parent_comm_id}_sub_{j}"
                                self.community_hierarchy[level][sub_comm_id] = list(sub_comm)
                    else:
                        # If no meaningful subdivision, keep parent community
                        self.community_hierarchy[level][parent_comm_id] = parent_nodes
                except Exception as e:
                    logger.warning(f"Subdivision failed for {parent_comm_id}: {e}")
                    self.community_hierarchy[level][parent_comm_id] = parent_nodes
        
        # Log hierarchy statistics
        for level, communities in self.community_hierarchy.items():
            logger.info(f"Level {level}: {len(communities)} communities, "
                       f"avg size: {np.mean([len(comm) for comm in communities.values()]):.1f}")
        
        return self.community_hierarchy

    # ==================== COMMUNITY SUMMARIZATION ====================

    def safe_json_parse(self, response_text: str) -> dict:
        """Safely parse JSON with fallback handling for community reports"""
        if not response_text or response_text.strip() == "":
            logger.warning("Empty response received from LLM")
            return self._create_fallback_community_data()
        
        # Clean the response text
        response_text = response_text.strip()
        
        try:
            # Try direct JSON parsing first
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed: {e}")
            
            try:
                # Try to extract JSON block from response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_part = response_text[json_start:json_end]
                    return json.loads(json_part)
            except json.JSONDecodeError:
                pass
            
            try:
                # Try to fix common JSON issues
                fixed_text = response_text.replace('``````', '')
                fixed_text = re.sub(r'^[^{]*', '', fixed_text)  # Remove text before first {
                fixed_text = re.sub(r'}[^}]*$', '}', fixed_text)  # Remove text after last }
                return json.loads(fixed_text)
            except:
                pass
            
            # Final fallback: create valid response
            logger.warning(f"All JSON parsing attempts failed. Response: {response_text[:200]}...")
            return self._create_fallback_community_data()

    def _create_fallback_community_data(self) -> dict:
        """Create fallback community data when JSON parsing fails"""
        return {
            "title": "Community Analysis",
            "summary": "This community contains interconnected entities in the court system with shared jurisdictional or administrative relationships.",
            "rating": 5.0,
            "rating_explanation": "Moderate impact community with standard court operations.",
            "findings": [
                {
                    "summary": "Court System Relationships",
                    "explanation": "This community represents courts and related entities that share jurisdictional or administrative connections within the legal system."
                }
            ]
        }


    
    async def generate_community_reports(self):
        """Generate community reports for all levels using Microsoft's approach"""
        logger.info("📝 Generating community reports...")
        
        # Generate reports bottom-up (leaf to root)
        for level in reversed(range(len(self.community_hierarchy))):
            communities = self.community_hierarchy[level]
            
            logger.info(f"Generating reports for level {level} ({len(communities)} communities)")
            
            # Process communities in batches to control API usage
            batch_size = 5
            for i in range(0, len(communities), batch_size):
                batch_communities = list(communities.items())[i:i+batch_size]
                
                tasks = [
                    self._generate_single_community_report(comm_id, nodes, level)
                    for comm_id, nodes in batch_communities
                ]
                
                batch_reports = await asyncio.gather(*tasks, return_exceptions=True)
                
                for j, report in enumerate(batch_reports):
                    if isinstance(report, Exception):
                        logger.error(f"❌ Error generating report: {report}")
                        continue
                    if report:
                        comm_id = batch_communities[j][0]
                        self.community_reports[comm_id] = report

    async def _generate_single_community_report(self, comm_id: str, nodes: List[str], level: int) -> Optional[CommunityReport]:
        """Generate a single community report using Microsoft's prompt"""
        
        # Gather community data
        entities_data = []
        relationships_data = []
        claims_data = []
        
        # Get entities in this community
        for node in nodes:
            for entity_id, entity in self.entities.items():
                if entity.name == node:
                    entities_data.append(f"id: {entity_id}, name: {entity.name}, type: {entity.type}, description: {entity.description}")
        
        # Get relationships within this community
        for rel_id, rel in self.relationships.items():
            if rel.source in nodes and rel.target in nodes:
                relationships_data.append(f"id: {rel_id}, source: {rel.source}, target: {rel.target}, description: {rel.description}")
        
        # Get claims about entities in this community
        for claim_id, claim in self.claims.items():
            if any(entity_name in nodes for entity_name in [claim.subject, claim.object]):
                claims_data.append(f"id: {claim_id}, subject: {claim.subject}, predicate: {claim.predicate}, object: {claim.object}, description: {claim.description}")
        
        # Prepare prompt data
        entities_text = "\n".join(entities_data[:50])  # Limit for token constraints
        relationships_text = "\n".join(relationships_data[:50])
        claims_text = "\n".join(claims_data[:30])
        
        # Generate report using Microsoft's prompt
        summary_prompt = self.get_community_summary_prompt()
        summary_chain = summary_prompt | self.llm
        
        try:
            response = await self.make_llm_call_with_rate_limiting(
                summary_chain,
                {
                    "entities": entities_text,
                    "relationships": relationships_text,
                    "claims": claims_text
                }
            )
            
            # Use safe JSON parsing instead of direct json.loads()
            response_data = self.safe_json_parse(response)
            
            # Create community report
            report = CommunityReport(
                community_id=comm_id,
                level=level,
                title=response_data.get("title", f"Community {comm_id}"),
                summary=response_data.get("summary", ""),
                rating=float(response_data.get("rating", 5.0)),
                rating_explanation=response_data.get("rating_explanation", ""),
                findings=response_data.get("findings", []),
                full_content=response,
                rank=len(nodes)
            )
            
            logger.info(f"✅ Generated report for {comm_id} (Level {level})")
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating report for {comm_id}: {e}")
            # Return a basic report instead of None
            return CommunityReport(
                community_id=comm_id,
                level=level,
                title=f"Community {comm_id}",
                summary="Unable to generate detailed summary",
                rating=5.0,
                rating_explanation="Default rating due to generation error",
                findings=[],
                full_content="",
                rank=len(nodes)
            )
        
    async def regenerate_community_reports_only(self):
        """Regenerate only the community reports without re-extracting entities"""
        logger.info("🔄 Regenerating community reports only...")
        
        # Clear existing reports
        self.community_reports = {}
        
        # Generate reports for all levels
        await self.generate_community_reports()
        
        logger.info(f"✅ Regenerated {len(self.community_reports)} community reports")

    # ==================== SEARCH CAPABILITIES ====================
    
    async def global_search(self, question: str, community_level: int = 0) -> str:
        """Perform global search using community summaries (Microsoft's approach)"""
        logger.info(f"🌍 Global search: {question}")
        
        # Get community reports for the specified level
        relevant_reports = []
        for comm_id, report in self.community_reports.items():
            if report.level == community_level:
                relevant_reports.append(report)
        
        if not relevant_reports:
            return "No community reports available for global search."
        
        # Sort by rank (importance)
        relevant_reports.sort(key=lambda x: x.rank, reverse=True)
        
        # Prepare reports text
        reports_text = []
        for i, report in enumerate(relevant_reports[:20]):  # Top 20 communities
            report_text = f"Report {i+1}:\nTitle: {report.title}\nSummary: {report.summary}\nRating: {report.rating}\nFindings: {json.dumps(report.findings)}"
            reports_text.append(report_text)
        
        # Use Microsoft's global search prompt
        global_prompt = self.get_global_search_prompt()
        global_chain = global_prompt | self.llm
        
        try:
            response = await self.make_llm_call_with_rate_limiting(
                global_chain,
                {
                    "reports": "\n\n".join(reports_text),
                    "question": question
                }
            )
            return response
        except Exception as e:
            logger.error(f"❌ Error in global search: {e}")
            return f"Error performing global search: {str(e)}"

    async def local_search(self, question: str, entity_name: str = None) -> str:
        """Perform local search around specific entities"""
        logger.info(f"🎯 Local search: {question}")
        
        if entity_name and entity_name in self.graph:
            # Search around specific entity
            neighbors = list(self.graph.neighbors(entity_name))
            context_nodes = [entity_name] + neighbors[:20]
        else:
            # Use a simple approach to find relevant entities
            context_nodes = list(self.graph.nodes())[:20]
        
        # Prepare context data
        context_data = []
        for node in context_nodes:
            node_data = self.graph.nodes[node]
            context_data.append(f"Entity: {node}, Type: {node_data.get('type', 'Unknown')}, Description: {node_data.get('description', '')}")
        
        # Use Microsoft's local search prompt
        local_prompt = self.get_local_search_prompt()
        local_chain = local_prompt | self.llm
        
        try:
            response = await self.make_llm_call_with_rate_limiting(
                local_chain,
                {
                    "context_data": "\n".join(context_data),
                    "question": question
                }
            )
            return response
        except Exception as e:
            logger.error(f"❌ Error in local search: {e}")
            return f"Error performing local search: {str(e)}"

    # ==================== VISUALIZATION ====================

    def remove_isolated_nodes(self):
        """Remove nodes with no connections from the graph"""
        # Get list of isolated nodes first (to avoid iteration error)
        isolated_nodes = list(nx.isolates(self.graph))
        
        if isolated_nodes:
            logger.info(f"🧹 Removing {len(isolated_nodes)} isolated nodes")
            
            # Remove from graph
            self.graph.remove_nodes_from(isolated_nodes)
            
            # Also remove from entities dict
            entities_to_remove = []
            for entity_id, entity in self.entities.items():
                if entity.name in isolated_nodes:
                    entities_to_remove.append(entity_id)
            
            for entity_id in entities_to_remove:
                del self.entities[entity_id]
            
            logger.info(f"✅ Graph now has {self.graph.number_of_nodes()} connected nodes")
        else:
            logger.info("No isolated nodes found")
        
        return len(isolated_nodes)

    
    def visualize_knowledge_graph(self, output_file: str = "optimized_graphrag.html", 
                             show_communities: bool = True, community_level: int = 0,
                             remove_isolates: bool = True):
        """Create comprehensive visualization of the knowledge graph"""
        logger.info(f"🎨 Creating visualization...")
        
        # Create working copy of graph
        viz_graph = self.graph.copy()
        
        # Remove isolated nodes if requested
        if remove_isolates:
            isolated = list(nx.isolates(viz_graph))
            viz_graph.remove_nodes_from(isolated)
            logger.info(f"🧹 Removed {len(isolated)} isolated nodes from visualization")
        
        # Create pyvis network
        net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="black")
        
        # Color mapping for entity types
        type_colors = {
            'PERSON': '#FF6B6B',
            'ORGANIZATION': '#4ECDC4', 
            'LOCATION': '#45B7D1',
            'EVENT': '#96CEB4',
            'CONCEPT': '#FFEAA7',
            'TECHNOLOGY': '#DDA0DD',
            'PRODUCT': '#98D8C8',
            'SERVICE': '#F7DC6F',
            'OTHER': '#AED6F1'
        }
        
        # Community colors for level 0
        community_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', 
                        '#99CCFF', '#FFB366', '#B3B3FF', '#66FFB2', '#FFD700']
        
        # Get community assignments if showing communities
        node_to_community = {}
        if show_communities and community_level in self.community_hierarchy:
            for comm_id, nodes in self.community_hierarchy[community_level].items():
                color_idx = hash(comm_id) % len(community_colors)
                for node in nodes:
                    if node in viz_graph:  # Only assign color if node exists in viz_graph
                        node_to_community[node] = community_colors[color_idx]
        
        # Add nodes
        for node in viz_graph.nodes():
            node_data = viz_graph.nodes[node]  # Use viz_graph instead of self.graph
            
            # Determine color
            if show_communities and node in node_to_community:
                color = node_to_community[node]
            else:
                color = type_colors.get(node_data.get('type', 'OTHER'), '#CCCCCC')
            
            # Create hover text
            hover_text = f"<b>{node}</b><br>"
            hover_text += f"Type: {node_data.get('type', 'Unknown')}<br>"
            hover_text += f"Description: {node_data.get('description', 'No description')[:200]}...<br>"
            hover_text += f"Connections: {viz_graph.degree(node)}"  # Use viz_graph degree
            
            # Add community info if available
            if show_communities:
                for comm_id, nodes in self.community_hierarchy.get(community_level, {}).items():
                    if node in nodes:
                        hover_text += f"<br>Community: {comm_id}"
                        break
            
            # Node size based on degree (use viz_graph)
            size = min(max(viz_graph.degree(node) * 3, 10), 50)
            
            net.add_node(
                node,
                label=node[:20] + "..." if len(node) > 20 else node,
                title=hover_text,
                color=color,
                size=size
            )
        
        # Add edges (use viz_graph)
        for source, target, edge_data in viz_graph.edges(data=True):
            hover_text = f"<b>{source}</b> → <b>{target}</b><br>"
            hover_text += f"Description: {edge_data.get('description', 'No description')}<br>"
            hover_text += f"Strength: {edge_data.get('strength', 'Unknown')}"
            
            net.add_edge(
                source,
                target,
                title=hover_text,
                width=max(edge_data.get('strength', 1) / 2, 1),
                color='#888888'
            )
        
        # Configure physics
        net.set_options("""
        var options = {
        "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
            "gravitationalConstant": -8000,
            "centralGravity": 0.3,
            "springLength": 200,
            "springConstant": 0.04,
            "damping": 0.09
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200
        }
        }
        """)
        
        # Save visualization
        net.save_graph(output_file)
        logger.info(f"✅ Visualization saved to {output_file}")
        
        # Try to open in browser
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(output_file))
        except:
            pass


    # ==================== MAIN PIPELINE ====================
    
    async def process_documents(self, documents: List[str], chunk_size: int = 600, batch_size: int = 10):
        """Complete Microsoft GraphRAG pipeline with cost optimization"""
        logger.info("🚀 Starting Optimized Microsoft GraphRAG pipeline...")
        
        # Step 1: Chunk documents
        self.text_units = self.chunk_documents(documents, chunk_size)
        
        # Step 2: Batch extract entities, relationships, and claims
        logger.info("🔍 Batch extracting entities, relationships, and claims...")
        
        all_entities, all_relationships, all_claims = await self.process_text_units_in_batches(
            self.text_units, batch_size=batch_size
        )
        
        # Step 3: Consolidate and store extractions
        logger.info("📝 Consolidating extractions...")
        
        # Consolidate entities (merge duplicates)
        entity_map = {}
        for i, entity in enumerate(all_entities):
            entity_id = f"entity_{i}"
            if entity.name in entity_map:
                # Merge descriptions
                existing_entity = self.entities[entity_map[entity.name]]
                existing_entity.description += f" {entity.description}"
            else:
                entity_map[entity.name] = entity_id
                self.entities[entity_id] = entity
        
        # Store relationships
        for i, relationship in enumerate(all_relationships):
            rel_id = f"relationship_{i}"
            self.relationships[rel_id] = relationship
        
        # Store claims
        for i, claim in enumerate(all_claims):
            claim_id = f"claim_{i}"
            self.claims[claim_id] = claim
        
        logger.info(f"✅ Extracted: {len(self.entities)} entities, {len(self.relationships)} relationships, {len(self.claims)} claims")
        
        # Step 4: Build knowledge graph
        self.build_knowledge_graph()
        
        # Step 5: Detect hierarchical communities
        self.detect_hierarchical_communities()
        
        # Step 6: Generate community reports
        await self.generate_community_reports()
        
        # Final cost summary
        estimated_cost = self.total_tokens_used * 0.00003  # Rough estimate for GPT-4o-mini
        logger.info(f"💰 Final cost summary: {self.api_calls_made} API calls, {self.total_tokens_used:,} tokens, ~${estimated_cost:.2f}")
        
        logger.info("🎉 Optimized GraphRAG pipeline completed!")

    # ==================== SAVE/LOAD ====================
    
    def save_system(self, filepath: str):
        """Save the complete GraphRAG system"""
        logger.info(f"💾 Saving GraphRAG system to {filepath}")
        
        data = {
            'text_units': [asdict(unit) for unit in self.text_units],
            'entities': {k: asdict(v) for k, v in self.entities.items()},
            'relationships': {k: asdict(v) for k, v in self.relationships.items()},
            'claims': {k: asdict(v) for k, v in self.claims.items()},
            'graph': nx.node_link_data(self.graph),
            'community_hierarchy': self.community_hierarchy,
            'community_reports': {k: asdict(v) for k, v in self.community_reports.items()},
            'total_tokens_used': self.total_tokens_used,
            'api_calls_made': self.api_calls_made,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info("✅ System saved successfully")

    def load_system(self, filepath: str):
        """Load a previously saved GraphRAG system"""
        logger.info(f"📂 Loading GraphRAG system from {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct objects
        self.text_units = [TextUnit(**unit) for unit in data['text_units']]
        self.entities = {k: Entity(**v) for k, v in data['entities'].items()}
        self.relationships = {k: Relationship(**v) for k, v in data['relationships'].items()}
        self.claims = {k: Claim(**v) for k, v in data['claims'].items()}
        self.graph = nx.node_link_graph(data['graph'])
        self.community_hierarchy = data['community_hierarchy']
        self.community_reports = {k: CommunityReport(**v) for k, v in data['community_reports'].items()}
        self.total_tokens_used = data.get('total_tokens_used', 0)
        self.api_calls_made = data.get('api_calls_made', 0)
        
        logger.info("✅ System loaded successfully")

# ==================== MAIN EXECUTION ====================

async def main():
    """Main execution function with cost optimization"""
    
    # Configuration
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
    
    # Initialize Optimized GraphRAG
    graphrag = OptimizedGraphRAG(
        azure_endpoint=AZURE_ENDPOINT,
        azure_key=AZURE_KEY,
        azure_deployment=AZURE_DEPLOYMENT,
        mistral_key=MISTRAL_KEY,
        max_concurrent_requests=8,  # Adjust based on your rate limits
        request_delay=0.1  # Small delay between requests
    )
    
    # Load documents (adapt this to your data source)
    documents = []

    # Option 1: Load from PDF
    pdf_path = "/Users/Viku/Datasets/Rag/Legal/PDF/41.pdf"  # Update with your PDF path
    if os.path.exists(pdf_path):
        print(f"📄 Loading PDF: {pdf_path}")
        from langchain_community.document_loaders import PyPDFLoader
        
        pdf_loader = PyPDFLoader(pdf_path)
        pdf_docs = pdf_loader.load()
        
        # Extract text from each page
        for doc in pdf_docs:
            documents.append(doc.page_content)
        
        print(f"✅ Loaded {len(pdf_docs)} pages from PDF")

    
    # # Option 2 Load from CSV (your court data)
    # csv_path = "/Users/Viku/Datasets/Rag/Legal/courts-2025-04-30.csv"  # Update with your path
    # if os.path.exists(csv_path):
    #     df = pd.read_csv(csv_path)
    #     df.columns.drop('notes', errors='ignore')  # Drop 'notes' if it exists
    #     # Combine relevant columns into text
    #     for _, row in df.iterrows():
    #         text_parts = []
    #         for col in ['short_name', 'full_name', 'jurisdiction', 'notes']:
    #             if col in df.columns and pd.notna(row[col]):
    #                 text_parts.append(f"{col}: {row[col]}")
    #         documents.append(". ".join(text_parts))
        
    #     # For testing, limit to first 100 documents
    #     documents = documents[:1500]  # Remove this line for full processing

    # # Option 3: Load multiple PDFs from directory
    # pdf_directory = "/Users/Viku/Datasets/Rag/Legal/PDFs/"
    # if os.path.exists(pdf_directory):
    #     print(f"📁 Loading PDFs from directory: {pdf_directory}")
    #     from langchain_community.document_loaders import PyPDFDirectoryLoader
        
    #     pdf_dir_loader = PyPDFDirectoryLoader(pdf_directory)
    #     pdf_docs = pdf_dir_loader.load()
        
    #     for doc in pdf_docs:
    #         documents.append(doc.page_content)
        
    #     print(f"✅ Loaded {len(pdf_docs)} pages from PDF directory")
    
    print(f"📚 Processing {len(documents)} documents...")
    
    # Check for saved system
    saved_file = "optimized_graphrag_system.pkl"
    
    if os.path.exists(saved_file):
        print("📂 Loading saved GraphRAG system...")
        graphrag.load_system(saved_file)
    else:
        print("🔨 Building new GraphRAG system...")
        await graphrag.process_documents(
            documents, 
            chunk_size=600, 
            batch_size=15  # Process 15 units concurrently
        )
        graphrag.save_system(saved_file)
    
    # Create visualizations
    print("🎨 Creating visualizations...")
    graphrag.visualize_knowledge_graph("optimized_graph.html", show_communities=True)
    
    # Interactive mode
    print("\n" + "="*60)
    print("🎮 OPTIMIZED GRAPHRAG - Interactive Mode")
    print("="*60)
    print("Commands:")
    print("• global <query> - Global search using community summaries")
    print("• local <query> [entity] - Local search around specific entity")
    print("• communities - Show community hierarchy")
    print("• stats - Show system statistics")
    print("• costs - Show cost breakdown")
    print("• viz - Create new visualization")
    print("• save <filename> - Save system state")
    print("• quit - Exit")
    
    while True:
        try:
            query = input("\n🔍 Enter command: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.startswith('global '):
                search_query = query[7:]
                result = await graphrag.global_search(search_query)
                print(f"\n🌍 **Global Search Results:**\n{result}")
            elif query.startswith('local '):
                parts = query[6:].split(' ', 1)
                search_query = parts[0]
                entity = parts[1] if len(parts) > 1 else None
                result = await graphrag.local_search(search_query, entity)
                print(f"\n🎯 **Local Search Results:**\n{result}")
            elif query == 'communities':
                print(f"\n🏗️ **Community Hierarchy:**")
                for level, communities in graphrag.community_hierarchy.items():
                    print(f"Level {level}: {len(communities)} communities")
                    for comm_id, nodes in list(communities.items())[:3]:
                        print(f"  {comm_id}: {len(nodes)} entities")
                    if len(communities) > 3:
                        print(f"  ... and {len(communities) - 3} more")
            elif query == 'stats':
                print(f"\n📊 **System Statistics:**")
                print(f"   • Text Units: {len(graphrag.text_units):,}")
                print(f"   • Entities: {len(graphrag.entities):,}")
                print(f"   • Relationships: {len(graphrag.relationships):,}")
                print(f"   • Claims: {len(graphrag.claims):,}")
                print(f"   • Graph Nodes: {graphrag.graph.number_of_nodes():,}")
                print(f"   • Graph Edges: {graphrag.graph.number_of_edges():,}")
                print(f"   • Community Reports: {len(graphrag.community_reports):,}")
                print(f"   • Hierarchy Levels: {len(graphrag.community_hierarchy)}")
            elif query == 'costs':
                estimated_cost = graphrag.total_tokens_used * 0.00003
                print(f"\n💰 **Cost Breakdown:**")
                print(f"   • API Calls Made: {graphrag.api_calls_made:,}")
                print(f"   • Total Tokens Used: {graphrag.total_tokens_used:,}")
                print(f"   • Estimated Cost: ~${estimated_cost:.2f}")
                print(f"   • Avg Tokens/Call: {graphrag.total_tokens_used/max(graphrag.api_calls_made,1):.0f}")
            elif query == 'viz':
                graphrag.visualize_knowledge_graph()
            elif query.startswith('save '):
                filename = query[5:].strip()
                graphrag.save_system(filename)
                print(f"✅ System saved to {filename}")
            else:
                print("❌ Unknown command. Try 'global <query>' or 'local <query>'")
                
        except KeyboardInterrupt:
            break
    
    print("👋 Thanks for using Optimized GraphRAG!")

if __name__ == "__main__":
    asyncio.run(main())
