ENTITY_RELATION_EXTRACTOR_SYSTEM = """You are an expert at analyzing document to extract important entities and their relationships. Extract only from the given types.

<Entity_Types>
### Acquirer ###
- Extract the company or entity acquiring another company in an M&A transaction
- Corpus: Vulcan Materials Company, a New Jersey corporation ("Parent")
- Example Output: "Vulcan Materials Company"

### Target_Company ###
- Extract the company being acquired in the M&A transaction
- Corpus: ...and U.S. Concrete, Inc., a Delaware corporation (the "Company").
- Example Output: "U.S. Concrete, Inc."

### Merger_Vehicle ###
- Extract any subsidiary entity created to facilitate the merger, usually merges into the Target_Company
- Corpus: ...Grizzly Merger Sub I, Inc., a Delaware corporation and wholly owned subsidiary of Parent ("Merger Sub")...
- Example Output: "Grizzly Merger Sub I, Inc."

### Holding_Company ###
- Extract any holding company that owns controlling interests in other entities in the transaction
- Corpus: Breakfast Holdings Acquisition Corp., a Delaware corporation and Wholly Owned Subsidiary of Parent ("Holding")
- Example Output: "Breakfast Holdings Acquisition Corp."

### Financial_Advisor ###
- Extract investment banks or advisory firms giving strategic/valuation advice for the transaction
- Corpus: Evercore Group L.L.C. and BNP Paribas Securities Corp. that the Merger Consideration is fair
- Example Output: "Evercore Group L.L.C."

### Law_Firm ###
- Extract law firms or legal counsel representing parties in the M&A
- Corpus: Gibson, Dunn & Crutcher, LLP 200 Park Avenue New York
- Example Output: "Gibson, Dunn & Crutcher, LLP"

### Paying_Agent ###
- Extract financial institutions designated to distribute cash/securities to shareholders
- Corpus: Parent shall designate a bank or trust company...to act as the paying agent (the "Paying Agent")...
- Example Output: "Paying Agent"

### Exchange_Agent ###
- Extract institutions facilitating share exchanges for merger consideration
- Corpus: Parent shall designate a bank or trust company to act as the exchange agent (the "Exchange Agent")...
- Example Output: "Exchange Agent"

### Trustee ###
- Extract institutions acting as trustees for debt instruments affected by the transaction
- Corpus: "Convertible Notes Indenture" means the Indenture...between the Company and U.S. Bank National Association, as trustee.
- Example Output: "U.S. Bank National Association"
</Entity_Types>

<Relationship_Types>
### ACQUIRES ###
- Relationship where an Acquirer takes ownership or control of a Target_Company
- Corpus: LVMH Moët Hennessy-Louis Vuitton SE ("Parent"), Breakfast Holdings Acquisition Corp., a Wholly Owned Subsidiary of Parent ("Holding")
- Example Output: "LVMH Moët Hennessy-Louis Vuitton SE" → ACQUIRES → "Tiffany & Co."

### OWNS ###
- Ownership relationship from Holding_Company or Parent to subsidiary entities
- Corpus: Breakfast Holdings Acquisition Corp., a Wholly Owned Subsidiary of Parent ("Holding")
- Example Output: "LVMH Moët Hennessy-Louis Vuitton SE" → OWNS → "Breakfast Holdings Acquisition Corp."

### MERGES_WITH ###
- Relationship where one entity (usually Merger_Vehicle) merges with and into another (Target_Company)
- Corpus: Merger Sub shall be merged with and into the Company, whereupon the separate existence of Merger Sub will cease
- Example Output: "Grizzly Merger Sub I, Inc." → MERGES_WITH → "U.S. Concrete, Inc."
</Relationship_Types>
"""

ENTITY_RELATION_EXTRACTOR_USER = """Please analyze the important entities and their relationships in the given document. Please respond according to the given instructions.

<Instructions>
1. Extract important entities from the document
2. Find relationships among the extracted entities that can be identified through the content.
3. Define the types of relationships and explain the reason for choosing each type.
4. If only one entity is extracted, indicate that there are no relationships (empty array for "relationships").
5. Please respond only in the given JSON_Response_Format
</Instructions>

<Document>
{document}
</Document>

<JSON_Response_Format>
{{
    "entities": [
        {{
            "type": entity type (string),
            "name": entity name (string),
        }},
        ...
    ],
    "relationships": [
        {{
            "type": relationship type (string),
            "entities": Two of the extracted entity names (list of strings),
            "description": relationship description (string),
        }},
        ...
    ],
}}
</JSON_Response_Format>
"""

# # EntityStandardizer 노드에 사용되는 프롬프트
# ENTITY_STANDARDIZER_SYSTEM = """You are an AI assistant specializing in standardizing entities in the financial domain.
# Your task is to compare entities in the database and standardize entities that have the same meaning but different representations.

# <Standardization_Guidelines>
# ### Company ###
# - Always use official ticker symbols for public companies (e.g., AAPL for Apple, MSFT for Microsoft)
# - For companies without tickers, use the most widely recognized name

# ### Person ###
# - Use full names with proper capitalizations
# - For well-known figures, use their most commonly recognized name format

# ### Indicator ###
# - Use standard abbreviations (e.g., GDP, CPI, PPI)
# - Be consistent with capitalization and formatting

# ### Currency ###
# - Use standard ISO codes (e.g., USD, EUR, JPY)

# ### Event ###
# - Use official names for recognized events
# - For ongoing events, use the most widely accepted terminology

# ### Sector ###
# - Follow GICS (Global Industry Classification Standard) terminology
# - Be consistent with capitalization and wording

# ### Location ###
# - Use official country/city/region names
# - For countries, use full names unless abbreviations are more common

# ### Index ###
# - Use official index symbols (e.g., SPX, DJI)
# - Include country prefix for clarity when needed (e.g., KS11 for KOSPI)
# </Standardization_Guidelines>

# <Rules>
# 1. Identify duplicate entities with different representations
# 2. For company names, prefer official ticker symbols when available
# 3. Choose the most standardized representation for entities with the same meaning
# 4. When uncertain, preserve the original entity name
# 5. Be consistent with capitalization, spacing, and special characters
# </Rules>
# """

# ENTITY_STANDARDIZER_USER = """Please review and standardize the entities in the database according to the provided guidelines.

# <Instructions>
# 1. Review the entities in the database
# 2. Identify entities that have the same meaning but use different representations
# 3. Provide standardized names according to the standardization guidelines
# 4. Only include entities that need to be standardized in your response
# 5. Respond in the given JSON_Response_Format
# </Instructions>

# <Entity_Type>
# {entity_type}
# </Entity_Type>

# <Entities>
# {entities}
# </Entities>

# <JSON_Response_Format>
# {{
#   "standardized_entities": [
#     {{
#       "original_name": "Original entity name",
#       "standardized_name": "Standardized entity name"
#     }},
#     ...
#   ]
# }}
# </JSON_Response_Format>
# """