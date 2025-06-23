import re 
import json 
from typing import List,Tuple,Dict
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma3:1b")

def chunk_text(text:str, chunk_size:int = 500) -> List[str]:
  """Chunking: Divide el texto en fragmentos manejables

  Args:
      text (str): texto para fragmentar
      chunk_size (int, optional): tamaño de fragmentos resultantes. Defaults to 500.

  Returns:
      List[str]: resultado de la operación
  """
  words = text.split()
  chunks = []
  current_chunk = []
  current_length = 0
  
  for word in words:
    current_chunk.append(word)
    current_length += len(word) + 1
    
    if current_length >= chunk_size:
      chunks.append(' '.join(current_chunk))
      current_chunk = []
      current_length = 0
  
  if current_chunk:
    chunks.append(' '.join(current_chunk))
  
  return chunks



def coreference_resolution(text:str) -> str:
  "Resuelve pronombres y referencias en el texto"
  prompt = f""" 
  Resolve all pronouns and coreferences in the following text. 
  Replace pronouns (he, she, it, they, etc.) with the actual entities they refer to.
  Make the text clear and unambiguous.
  
  Text: {text}
  
  Resolved text:
  """
  resolved = llm.invoke(prompt)
  return resolved.strip()



def named_entity_recognition(text:str) -> List[Dict]:
  "Extrae entidades nombradas del texto"
  prompt = f"""
  Extract all named entities from the following text.
  Identify PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT, and other important entities.
  
  Format your response as a JSON list of objects with 'entity' and 'type' fields.
  
  Example format:
  [
    {{"entity": "John Smith", "type": "PERSON"}},
    {{"entity": "Microsoft", "type": "ORGANIZATION"}},
    {{"entity": "New York", "type": "LOCATION"}}
  ]
  
  Text: {text}
  
  Entities (JSON format only):
  """
  response = llm.invoke(prompt)
  try:
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if json_match:
      entities = json.loads(json_match.group())
      return entities
  except:
    pass 
  
  # análisis alternativo si JSON falla
  entities = []
  lines = response.split('\n')
  for line in lines:
    if '"entity"' in line and '"type"' in line:
      try:
        entity_match = re.search(r'"entity":\s*"([^"]+)"', line)
        type_match = re.search(r'"type":\s*"([^"]+)"', line)
        if entity_match and type_match:
          entities.append({
            "entity": entity_match.group(1),
            "type": type_match.group(1)
          })
      except:
        continue
  return entities




def relationship_extraction(text:str, entities:List[Dict]) -> List[Tuple[str,str,str]]:
  "Extrae relaciones entre entidades"
  entity_list = [e['entity'] for e in entities]
  entity_str = ', '.join(entity_list)
  
  prompt = f""" 
  Given the following text and list of entities, extract relationships between entities.
  
  Text: {text}
  
  Entities: {entity_str}
  
  Extract relationships in the format: (Entity1, Relationship, Entity2)
  
  Examples:
  - (John Smith, works_for, Microsoft)
  - (Microsoft, located_in, Seattle)
  - (John Smith, lives_in, New York)
  
  Focus on meaningful relationships like:
  - works_for, employed_by
  - located_in, based_in
  - founded_by, created_by
  - part_of, member_of
  - leads, manages
  - interested_in, likes
  - related_to, connected_to
  
  Extract relationships (one per line):
  """
  response = llm.invoke(prompt)
  
  # parse relationships from response 
  relationships = []
  lines = response.split('\n')
  
  for line in lines:
    line = line.strip()
    # Look for patterns like (Entity1, relation, Entity2)
    match = re.search(r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)', line)
    if match:
      entity1 = match.group(1).strip()
      relation = match.group(2).strip()
      entity2 = match.group(3).strip()
      relationships.append((entity1, relation, entity2))
    # Also look for patterns like Entity1 -> relation -> Entity2
    elif '->' in line:
      parts = line.split('->')
      if len(parts) == 3:
        entity1 = parts[0].strip()
        relation = parts[1].strip()
        entity2 = parts[2].strip()
        relationships.append((entity1, relation, entity2))
  
  return relationships




def entity_disambiguation(entities:List[Dict]) -> List[Dict]:
  "Desambigua y normaliza las entidades"
  if not entities:
    return entities
  
  entity_str = '\n'.join([f"- {e['entity']} ({e['type']})" for e in entities])
  prompt = f"""
  Disambiguate and normalize the following entities. 
  Remove duplicates, resolve different names for the same entity, and standardize the names.
  
  Entities:
  {entity_str}
  
  Return the disambiguated entities in the same JSON format:
  [
    {{"entity": "normalized_name", "type": "TYPE"}}
  ]
  
  Disambiguated entities (JSON format only):
  """
  
  response = llm.invoke(prompt)
  
  try:
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if json_match:
      disambiguated = json.loads(json_match.group())
      return disambiguated
  except:
    pass
  
  # devuelve las entidades originales si la desambiguación falla
  return entities



def extract_triples(text:str) -> List[Tuple[str,str,str]]:
  """Extrae tripletas de un fragmento de texto (chunk)

  Args:
      text (str): Fragmento de texto de entrada

  Returns:
      List[Tuple[str,str,str]]: Lista de tripletas (entity_1, relationship, entity_2)
  """
  print(f"Processing text chunk: {text[:100]}...")
  
  # Step 1: Coreference Resolution
  resolved_text = coreference_resolution(text)
  print("Coreference resolution completed")
  
  # Step 2: Named Entity Recognition
  entities = named_entity_recognition(resolved_text)
  print(f"Found {len(entities)} entities")
  
  # Step 3: Entity Disambiguation
  entities = entity_disambiguation(entities)
  print(f"Disambiguated to {len(entities)} unique entities")
  
  # Step 4: Relationship Extraction
  relationships = relationship_extraction(resolved_text, entities)
  print(f"Extracted {len(relationships)} relationships")
  
  return relationships

def process_document(document_text:str, verbose:bool = False) -> List[Tuple[str,str,str]]:
  "Chunk documents + extract triples"
  chunks = chunk_text(document_text)
  
  all_triples = []
  if verbose: print(f"Processing document with {len(chunks)} chunks")
  for i, chunk in enumerate(chunks):
    if verbose: print(f"Processing Chunk {i+1}/{len(chunks)}")
    triples = extract_triples(chunk)
    all_triples.extend(triples)
  
  # remover duplicados
  unique_triples = list(set(all_triples))
  print(f"\nExtracted {len(unique_triples)} unique triples from document")
  
  return unique_triples