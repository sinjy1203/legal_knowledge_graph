import json
from langchain_core.prompts import ChatPromptTemplate
from generate_knowledge_graph.utils import *
from logger import setup_logger
from langgraph.types import Command
from generate_knowledge_graph.utils.callback import BatchCallback

logger = setup_logger()

GRAPH_SCHEMA = {
  "entity": [
    {
      "type": "Acquirer",
      "description": "The company or entity that is purchasing or taking control of another company in a merger or acquisition transaction",
      "examples": [
        {
          "corpus": "Vulcan Materials Company, a New Jersey corporation (\"Parent\")",
          "extracted_entities": [
            "Vulcan Materials Company"
          ]
        }
      ]
    },
    {
      "type": "TargetCompany",
      "description": "The company being acquired, purchased, or merged into another entity in an M&A transaction",
      "examples": [
        {
          "corpus": "...and U.S. Concrete, Inc., a Delaware corporation (the \"Company\").",
          "extracted_entities": [
            "U.S. Concrete, Inc."
          ]
        }
      ]
    },
    {
      "type": "MergerVehicle",
      "description": "A subsidiary entity created specifically to facilitate the merger transaction, which typically merges with and into the target company",
      "examples": [
        {
          "corpus": "...Grizzly Merger Sub I, Inc., a Delaware corporation and wholly owned subsidiary of Parent (\"Merger Sub\")...",
          "extracted_entities": [
            "Grizzly Merger Sub I, Inc."
          ]
        }
      ]
    },
    {
      "type": "HoldingCompany",
      "description": "A parent company that owns controlling interests in other companies and is used to structure complex acquisition transactions",
      "examples": [
        {
          "corpus": "Breakfast Holdings Acquisition Corp., a Delaware corporation and Wholly Owned Subsidiary of Parent (\"Holding\")",
          "extracted_entities": [
            "Breakfast Holdings Acquisition Corp."
          ]
        }
      ]
    },
    {
      "type": "FinancialAdvisor",
      "description": "Investment banks or financial advisory firms that provide valuation opinions, strategic advice, and facilitate M&A transactions",
      "examples": [
        {
          "corpus": "Evercore Group L.L.C. and BNP Paribas Securities Corp. that the Merger Consideration is fair",
          "extracted_entities": [
            "Evercore Group L.L.C."
          ]
        }
      ]
    },
    {
      "type": "LawFirm",
      "description": "Legal counsel representing parties in M&A transactions, providing legal advice and drafting transaction documents",
      "examples": [
        {
          "corpus": "Gibson, Dunn & Crutcher, LLP 200 Park Avenue New York",
          "extracted_entities": [
            "Gibson, Dunn & Crutcher, LLP"
          ]
        }
      ]
    },
    {
      "type": "PayingAgent",
      "description": "A financial institution designated to handle the exchange of cash and securities to target company shareholders in an M&A transaction",
      "examples": [
        {
          "corpus": "Parent shall designate a bank or trust company...to act as the paying agent (the \"Paying Agent\")...",
          "extracted_entities": [
            "Paying Agent"
          ]
        }
      ]
    },
    {
      "type": "ExchangeAgent",
      "description": "A financial institution that facilitates the exchange of target company shares for merger consideration (cash and/or acquirer shares)",
      "examples": [
        {
          "corpus": "Parent shall designate a bank or trust company to act as the exchange agent (the \"Exchange Agent\")...",
          "extracted_entities": [
            "Exchange Agent"
          ]
        }
      ]
    },
    {
      "type": "Trustee",
      "description": "A financial institution serving as trustee for debt instruments (bonds, notes) that may be affected by the M&A transaction",
      "examples": [
        {
          "corpus": "\"Convertible Notes Indenture\" means the Indenture...between the Company and U.S. Bank National Association, as trustee.",
          "extracted_entities": [
            "U.S. Bank National Association"
          ]
        }
      ]
    },
    {
      "type": "KnowledgeDefinition",
      "description": "Legal standard defining what constitutes \"knowledge\" for representation and warranty purposes, typically limited to specific individuals' actual knowledge",
      "examples": [
        {
          "corpus": "actual knowledge, after reasonable inquiry",
          "extracted_entities": [
            "Extended Stay America"
          ]
        }
      ]
    },
    {
      "type": "MaterialAdverseEffectDefinition",
      "description": "Legal definition of changes or events significant enough to materially impact a company's business, allowing deal termination or price adjustment",
      "examples": [
        {
          "corpus": "material adverse effect on the business, condition",
          "extracted_entities": [
            "Tiffany & Co."
          ]
        }
      ]
    },
    {
      "type": "CompanyMaterialAdverseEffect",
      "description": "Specific definition of material adverse effect applicable to the target company, with detailed carve-outs for general economic conditions",
      "examples": [
        {
          "corpus": "materially adverse to the business, results of operations",
          "extracted_entities": [
            "Extended Stay America"
          ]
        }
      ]
    },
    {
      "type": "AcquisitionProposalDefinition",
      "description": "Legal definition of competing takeover proposals that trigger no-shop restrictions and disclosure obligations in M&A agreements",
      "examples": [
        {
          "corpus": "any offer, proposal...relating to: (a) acquisition of more than 15% of outstanding voting securities, (b) merger, consolidation",
          "extracted_entities": [
            "Acquisition Proposal"
          ]
        }
      ]
    },
    {
      "type": "SuperiorProposalDefinition",
      "description": "Legal definition of a competing proposal that is more favorable to shareholders, allowing target board to change recommendation or terminate agreement",
      "examples": [
        {
          "corpus": "bona fide, written Acquisition Proposal...more favorable to Company Stockholders",
          "extracted_entities": [
            "Superior Proposal"
          ]
        }
      ]
    },
    {
      "type": "AcceptableConfidentialityAgreement",
      "description": "Standard for confidentiality agreements that must be signed before providing due diligence information to potential competing bidders",
      "examples": [
        {
          "corpus": "confidentiality agreement with terms no less favorable",
          "extracted_entities": [
            "Acceptable Confidentiality Agreement"
          ]
        }
      ]
    },
    {
      "type": "ChangeOfRecommendation",
      "description": "Actions by target company board that constitute withdrawal or modification of their recommendation in favor of the merger agreement",
      "examples": [
        {
          "corpus": "adopt, approve, publicly endorse any Acquisition Proposal",
          "extracted_entities": [
            "Change of Recommendation"
          ]
        }
      ]
    },
    {
      "type": "MergerConsideration",
      "description": "The price and form of payment (cash, stock, or mixed) that target company shareholders receive in exchange for their shares",
      "examples": [
        {
          "corpus": "each share shall be converted into the right to receive $131.50 in cash, without interest",
          "extracted_entities": [
            "Tiffany & Co."
          ]
        }
      ]
    },
    {
      "type": "TerminationFee",
      "description": "Penalty fee paid by target company to acquirer if the deal is terminated under specific circumstances, designed to deter competing bids",
      "examples": [
        {
          "corpus": "The Company shall pay to Parent a termination fee of $575,000,000 (the \"Company Termination Fee\") if this Agreement is terminated and within twelve months the Company enters into any Alternative Acquisition Agreement",
          "extracted_entities": [
            "Tiffany & Co."
          ]
        }
      ]
    },
    {
      "type": "HSRApproval",
      "description": "Required antitrust clearance from U.S. federal agencies under the Hart-Scott-Rodino Act for transactions above certain size thresholds",
      "examples": [
        {
          "corpus": "HSR Act and other requisite clearances under Antitrust Laws",
          "extracted_entities": [
            "HSR Act"
          ]
        }
      ]
    },
    {
      "type": "AntitrustApproval",
      "description": "Required regulatory clearances from competition authorities to ensure the transaction doesn't violate antitrust laws",
      "examples": [
        {
          "corpus": "requisite clearances under other applicable Antitrust Laws",
          "extracted_entities": [
            "Antitrust_Authorities"
          ]
        }
      ]
    },
    {
      "type": "SECFilingRequirements",
      "description": "Required filings and approvals from the Securities and Exchange Commission for public company transactions involving securities issuance",
      "examples": [
        {
          "corpus": "Registration Statement effective under Securities Act",
          "extracted_entities": [
            "SEC"
          ]
        }
      ]
    },
    {
      "type": "CompanyStockholderApproval",
      "description": "Required vote by target company shareholders to approve the merger agreement and authorize the transaction",
      "examples": [
        {
          "corpus": "affirmative vote of majority of outstanding shares",
          "extracted_entities": [
            "Company_Stockholders"
          ]
        }
      ]
    },
    {
      "type": "NoLegalProhibition",
      "description": "Condition ensuring no court orders, injunctions, or laws prohibit the completion of the merger transaction",
      "examples": [
        {
          "corpus": "No Governmental Entity shall have enacted any Law that prohibits the Merger",
          "extracted_entities": [
            "No Legal Prohibition"
          ]
        }
      ]
    },
    {
      "type": "NoShopProvision",
      "description": "Contractual restriction preventing target company from soliciting or encouraging competing acquisition proposals during the deal process",
      "examples": [
        {
          "corpus": "shall not initiate, solicit any Acquisition Proposal",
          "extracted_entities": [
            "Extended Stay America"
          ]
        }
      ]
    },
    {
      "type": "FiduciaryOut",
      "description": "Exception allowing target company board to change recommendation or terminate agreement when required by fiduciary duties to shareholders",
      "examples": [
        {
          "corpus": "bona fide written Acquisition Proposal",
          "extracted_entities": [
            "U.S. Concrete, Inc."
          ]
        }
      ]
    },
    {
      "type": "FiduciaryTerminationRight",
      "description": "Target company's right to terminate the merger agreement to accept a superior proposal, typically requiring payment of termination fee",
      "examples": [
        {
          "corpus": "by the Company in order to effect a Change of Recommendation and...enter into a definitive agreement providing for a Superior Proposal",
          "extracted_entities": [
            "Slack Technologies, Inc."
          ]
        }
      ]
    },
    {
      "type": "TailProvision",
      "description": "Mechanism extending termination fee obligations for a specified period after deal termination if target enters alternative transaction",
      "examples": [
        {
          "corpus": "within twelve months of termination, an Acquisition Proposal is consummated...then the Company shall pay to Parent a fee of nine hundred million dollars ($900,000,000)",
          "extracted_entities": [
            "Slack Technologies, Inc."
          ]
        }
      ]
    },
    {
      "type": "OrdinaryCourseCovenant",
      "description": "Contractual obligation for target company to operate its business in the ordinary course during the period between signing and closing",
      "examples": [
        {
          "corpus": "the Company shall conduct its business in all material respects in the ordinary course consistent with past practice",
          "extracted_entities": [
            "U.S. Concrete, Inc."
          ]
        }
      ]
    },
    {
      "type": "NegativeCovenant",
      "description": "Contractual restrictions on specific actions target company cannot take without acquirer consent during the interim period",
      "examples": [
        {
          "corpus": "shall not...directly or indirectly: (i) amend, modify...organizational documents; (ii) authorize, declare...any dividends...",
          "extracted_entities": [
            "U.S. Concrete, Inc."
          ]
        }
      ]
    },
    {
      "type": "ConsiderationStructure",
      "description": "The specific terms and structure of payment to target shareholders, including cash, stock, or combination thereof",
      "examples": [
        {
          "corpus": "Each share of Company Common Stock...shall be converted into (A) 0.0776 (the \"Exchange Ratio\") fully paid shares of Parent Common Stock, and (B) the right to receive $26.79 in cash",
          "extracted_entities": [
            "Slack Technologies, Inc."
          ]
        }
      ]
    },
    {
      "type": "TransactionType",
      "description": "The legal structure used to complete the acquisition, such as one-step merger, two-step merger, or tender offer followed by merger",
      "examples": [
        {
          "corpus": "Merger Subsidiary has agreed to commence a tender offer (the \"Offer\") to purchase...shares of common stock of the Company, at a price per share of $22.00",
          "extracted_entities": [
            "The Michaels Companies, Inc."
          ]
        }
      ]
    },
    {
      "type": "ExchangeRatio",
      "description": "The ratio determining how many acquirer shares target shareholders receive for each target share in a stock-for-stock transaction",
      "examples": [
        {
          "corpus": "0.0776 (the \"Exchange Ratio\") fully paid shares of Parent Common Stock",
          "extracted_entities": [
            "Slack Technologies, Inc."
          ]
        }
      ]
    },
    {
      "type": "MinimumCondition",
      "description": "Minimum ownership threshold that must be achieved in a tender offer before the acquirer is obligated to purchase tendered shares",
      "examples": [
        {
          "corpus": "there shall have been validly tendered...that number of Shares that...represent at least a majority of the Shares outstanding",
          "extracted_entities": [
            "The Michaels Companies, Inc."
          ]
        }
      ]
    },
    {
      "type": "CovenantCompliance",
      "description": "Requirement that target company has complied with all interim period covenants as a condition to closing the transaction",
      "examples": [
        {
          "corpus": "The Company shall have performed...in all material respects with the obligations...required under the Agreement",
          "extracted_entities": [
            "Slack Technologies, Inc."
          ]
        }
      ]
    },
    {
      "type": "NoMaterialAdverseEffect",
      "description": "Requirement that no material adverse effect has occurred to target company as a condition to completing the merger",
      "examples": [
        {
          "corpus": "there has not been any Effect...that has had, or would reasonably be expected to have, a Company Material Adverse Effect.",
          "extracted_entities": [
            "Cubic Corporation"
          ]
        }
      ]
    },
    {
      "type": "RepresentationAccuracy",
      "description": "Requirement that target company's representations and warranties remain true and correct as of closing date",
      "examples": [
        {
          "corpus": "The representations and warranties...shall be true and correct...except where the failure...has not had and would not reasonably be expected to have a Company Material Adverse Effect.",
          "extracted_entities": [
            "W. R. Grace & Co."
          ]
        }
      ]
    },
    {
      "type": "OrdinaryCourseOperations",
      "description": "Target company's commitment to operate business normally during interim period to preserve business value until closing",
      "examples": [
        {
          "corpus": "conduct operations in ordinary course of business",
          "extracted_entities": [
            "Inovalon Holdings, Inc."
          ]
        }
      ]
    },
    {
      "type": "ProhibitedActions",
      "description": "Specific corporate actions target company is prohibited from taking during interim period without acquirer consent",
      "examples": [
        {
          "corpus": "shall not amend organizational documents or authorize dividends",
          "extracted_entities": [
            "Inovalon Holdings, Inc."
          ]
        }
      ]
    },
    {
      "type": "SpecificRestrictions",
      "description": "Detailed restrictions on specific business activities like employee compensation changes or new acquisitions during interim period",
      "examples": [
        {
          "corpus": "shall not increase compensation or grant severance",
          "extracted_entities": [
            "Slack Technologies, Inc."
          ]
        }
      ]
    },
    {
      "type": "SpecificPerformance",
      "description": "Contractual provision allowing parties to seek court orders compelling performance of merger agreement obligations",
      "examples": [
        {
          "corpus": "The parties agree that irreparable damage would occur...the parties shall be entitled to an injunction...specific performance or other equitable relief",
          "extracted_entities": [
            "Magic AcquireCo, Inc."
          ]
        }
      ]
    },
    {
      "type": "TerminationFeeTriggers",
      "description": "Specific events or circumstances that trigger obligation to pay termination fee to protect acquirer's deal investment",
      "examples": [
        {
          "corpus": "If the Company terminates this Agreement...the Company shall pay to Parent the Termination Fee.",
          "extracted_entities": [
            "Slack Technologies, Inc."
          ]
        }
      ]
    },
    {
      "type": "DamagesLimitations",
      "description": "Contractual caps on monetary damages available to parties, typically limiting remedies to termination fees in certain circumstances",
      "examples": [
        {
          "corpus": "except the right to seek monetary damages for fraud...or for willful breach...the Termination Fee...shall be the sole and exclusive monetary remedy available",
          "extracted_entities": [
            "Merger Agreement"
          ]
        }
      ]
    }
  ],
  "relationship": [
    {
      "type": "Acquires",
      "description": "Relationship indicating one company is acquiring ownership or control of another company through the merger or acquisition transaction",
      "examples": [
        {
          "corpus": "LVMH Moët Hennessy-Louis Vuitton SE (\"Parent\"), Breakfast Holdings Acquisition Corp., a Wholly Owned Subsidiary of Parent (\"Holding\")",
          "source_entity": "LVMH Moët Hennessy-Louis Vuitton SE",
          "source_type": "Acquirer",
          "target_entity": "Tiffany & Co.",
          "target_type": "TargetCompany"
        }
      ]
    },
    {
      "type": "Owns",
      "description": "Ownership relationship between parent companies and their subsidiaries, often used to structure complex acquisition transactions",
      "examples": [
        {
          "corpus": "Breakfast Holdings Acquisition Corp., a Wholly Owned Subsidiary of Parent (\"Holding\")",
          "source_entity": "LVMH Moët Hennessy-Louis Vuitton SE",
          "source_type": "Acquirer",
          "target_entity": "Breakfast Holdings Acquisition Corp.",
          "target_type": "HoldingCompany"
        }
      ]
    },
    {
      "type": "MergesWith",
      "description": "Legal relationship where one entity merges with and into another, typically with one entity surviving and the other ceasing to exist",
      "examples": [
        {
          "corpus": "Merger Sub shall be merged with and into the Company, whereupon the separate existence of Merger Sub will cease",
          "source_entity": "Grizzly Merger Sub I, Inc.",
          "source_type": "MergerVehicle",
          "target_entity": "U.S. Concrete, Inc.",
          "target_type": "TargetCompany"
        }
      ]
    },
    {
      "type": "Advises",
      "description": "Professional advisory relationship where financial advisors provide valuation opinions, strategic advice, and transaction facilitation services",
      "examples": [
        {
          "corpus": "The Company Board has received an opinion of Qatalyst Partners LP that the Merger Consideration...is fair, from a financial point of view",
          "source_entity": "Qatalyst Partners LP",
          "source_type": "FinancialAdvisor",
          "target_entity": "Slack Technologies, Inc.",
          "target_type": "TargetCompany"
        }
      ]
    },
    {
      "type": "Represents",
      "description": "Legal representation relationship where law firms provide legal counsel and services to parties in M&A transactions",
      "examples": [
        {
          "corpus": "if to the Company: Slack Technologies, Inc...with copies to: Latham & Watkins LLP",
          "source_entity": "Latham & Watkins LLP",
          "source_type": "LawFirm",
          "target_entity": "Slack Technologies, Inc.",
          "target_type": "TargetCompany"
        }
      ]
    },
    {
      "type": "DefinedIn",
      "description": "Relationship indicating where legal terms and definitions are specifically defined within merger agreement documents",
      "examples": [
        {
          "corpus": "\"Knowledge\" will be deemed to be the actual knowledge of...the individuals set forth on Section 1.1(a) of the Company Disclosure Letter",
          "source_entity": "Knowledge",
          "source_type": "KnowledgeDefinition",
          "target_entity": "Slack Technologies, Inc.",
          "target_type": "TargetCompany"
        }
      ]
    },
    {
      "type": "CrossReferences",
      "description": "Internal document references linking related provisions within merger agreements to create coherent legal framework",
      "examples": [
        {
          "corpus": "the Company may...cause the Company to terminate this Agreement pursuant to and in accordance with Section 8.1(h)",
          "source_entity": "Section 5.3(d)",
          "source_type": "FiduciaryOut",
          "target_entity": "Section 8.1(h)",
          "target_type": "FiduciaryTerminationRight"
        }
      ]
    },
    {
      "type": "IncorporatesByReference",
      "description": "Legal mechanism incorporating external documents or schedules into the main merger agreement by reference",
      "examples": [
        {
          "corpus": "except as set forth in Section 5.1 of the Company Disclosure Letter",
          "source_entity": "Slack_Merger_Agreement",
          "source_type": "MergerConsideration",
          "target_entity": "Company Disclosure Letter",
          "target_type": "OrdinaryCourseCovenant"
        }
      ]
    },
    {
      "type": "GovernedBy",
      "description": "Relationship indicating which state or jurisdiction's laws govern the interpretation and enforcement of the merger agreement",
      "examples": [
        {
          "corpus": "shall be governed by and construed in accordance with the laws of the State of Delaware",
          "source_entity": "Agreement",
          "source_type": "MergerConsideration",
          "target_entity": "Delaware",
          "target_type": "HSRApproval"
        }
      ]
    },
    {
      "type": "SubjectTo",
      "description": "Conditional relationship where transaction completion depends on satisfaction of specific regulatory or other conditions",
      "examples": [
        {
          "corpus": "The waiting period under the HSR Act...shall have expired or terminated",
          "source_entity": "Slack_Acquisition",
          "source_type": "ConsiderationStructure",
          "target_entity": "HSR_Act_Approval",
          "target_type": "HSRApproval"
        }
      ]
    },
    {
      "type": "RequiresApprovalFrom",
      "description": "Relationship indicating which parties must provide formal approval for the transaction to proceed legally",
      "examples": [
        {
          "corpus": "The affirmative vote of the holders of a majority of the outstanding Company Common Stock...is the only vote necessary",
          "source_entity": "Slack_Acquisition",
          "source_type": "ConsiderationStructure",
          "target_entity": "Company_Stockholders",
          "target_type": "CompanyStockholderApproval"
        }
      ]
    },
    {
      "type": "Pays",
      "description": "Financial relationship indicating payment obligations from acquirer to target shareholders as merger consideration",
      "examples": [
        {
          "corpus": "Each share of Company Common Stock shall be converted into...0.0776...shares of Parent Common Stock...and...the right to receive $26.79 in cash",
          "source_entity": "salesforce.com, inc.",
          "source_type": "Acquirer",
          "target_entity": "Slack Technologies, Inc. Stockholders",
          "target_type": "TargetCompany"
        }
      ]
    },
    {
      "type": "TriggersFee",
      "description": "Conditional relationship where specific events or breaches trigger obligation to pay termination or other fees",
      "examples": [
        {
          "corpus": "within twelve months of termination, an Acquisition Proposal is consummated...then the Company shall pay to Parent a fee of nine hundred million dollars ($900,000,000)",
          "source_entity": "Tail_Provision_Breach",
          "source_type": "TailProvision",
          "target_entity": "Termination_Fee_Payment",
          "target_type": "CompanyTerminationFee"
        }
      ]
    },
    {
      "type": "TriggersIf",
      "description": "Conditional logic relationship where specific actions automatically trigger other consequences or obligations",
      "examples": [
        {
          "corpus": "If the Company terminates this Agreement...the Company shall pay to Parent the Termination Fee.",
          "source_entity": "Superior_Proposal_Termination",
          "source_type": "FiduciaryTerminationRight",
          "target_entity": "Termination_Fee_Payment",
          "target_type": "CompanyTerminationFee"
        }
      ]
    },
    {
      "type": "ExceptWhen",
      "description": "Exception relationship allowing deviation from general rules under specific circumstances, often related to fiduciary duties",
      "examples": [
        {
          "corpus": "the Company Board may make a Change of Recommendation...if the Company Board has determined...that the failure to take such action would...violate the directors' fiduciary duties",
          "source_entity": "No_Shop_Clause",
          "source_type": "NoShopProvision",
          "target_entity": "Fiduciary_Duties_Violation",
          "target_type": "FiduciaryOut"
        }
      ]
    },
    {
      "type": "SubjectToCondition",
      "description": "Conditional relationship where actions or closing depends on satisfaction of specific conditions or performance standards",
      "examples": [
        {
          "corpus": "The Company shall have performed...in all material respects with the obligations...required...at or prior to the Closing.",
          "source_entity": "Closing",
          "source_type": "CovenantCompliance",
          "target_entity": "Covenant_Compliance",
          "target_type": "CovenantCompliance"
        }
      ]
    }
  ]
}

ENTITY_TYPES = [entity["type"] for entity in GRAPH_SCHEMA["entity"]]
RELATIONSHIP_TYPES = [relationship["type"] for relationship in GRAPH_SCHEMA["relationship"]]

# SYSTEM_TEMPLATE = """You are an expert at analyzing document to extract important entities and their relationships. Extract only from the given graph schema.

# <Graph_Schema>
# {graph_schema}
# </Graph_Schema>
# """

# USER_TEMPLATE = """Please analyze the important entities and their relationships in the given document. Please respond according to the given instructions.

# <Instructions>
# 1. Extract important entities from the document
# 2. Find relationships among the extracted entities that can be identified through the content.
# 3. Define the types of relationships and explain the reason for choosing each type.
# 4. If only one entity is extracted, indicate that there are no relationships (empty array for "relationships").
# 5. Please respond only in the given JSON_Response_Format
# </Instructions>

# <Document>
# {document}
# </Document>

# <JSON_Response_Format>
# {{
#     "entities": [
#         {{
#             "type": entity type (string),
#             "name": entity name (string),
#         }},
#         ...
#     ],
#     "relationships": [
#         {{
#             "type": relationship type (string),
#             "source_entity": source entity name (string),
#             "source_type": source entity type (string),
#             "target_entity": target entity name (string),
#             "target_type": target entity type (string),
#             "description": relationship description (string),
#         }},
#         ...
#     ],
# }}
# </JSON_Response_Format>
# """


SYSTEM_TEMPLATE = """You are an expert at analyzing documents to extract important entities and their relationships. You must extract only entities and relationship types that exactly match those defined in the provided <Entity_Types> and <Relationship_Types>.

<Entity_Types>
{entity_types}
</Entity_Types>

<Relationship_Types>
{relationship_types}
</Relationship_Types>

Important:
- Do NOT generate or infer any entity or relationship types that are not explicitly listed in the <Entity_Types> and <Relationship_Types>.
- Only use the exact "type" values provided in the <Entity_Types> and <Relationship_Types> for both entities and relationships.
"""

USER_TEMPLATE = """Please analyze the important entities and their relationships in the given document. Only extract types that are explicitly defined in the provided graph schema. Follow the instructions carefully and respond strictly in the given format.

<Instructions>
1. Extract important entities from the document using only the entity types defined in the <Entity_Types>.
2. Identify relationships among the extracted entities using only the relationship types from the <Relationship_Types>.
3. For each relationship, specify its type and explain briefly why that relationship applies based on the content.
4. If only one entity is extracted, indicate that no relationships exist (return an empty array for "relationships").
5. Do NOT invent or use entity/relationship types that are not in the <Entity_Types> and <Relationship_Types>.
6. Respond only in the JSON format specified below.
</Instructions>

<Document>
{document}
</Document>

<JSON_Response_Format>
{{
    "entities": [
        {{
            "type": entity type (string, must match <Entity_Types>),
            "name": entity name (string)
        }},
        ...
    ],
    "relationships": [
        {{
            "type": relationship type (string, must match <Relationship_Types>),
            "source_entity": source entity name (string),
            "source_type": source entity type (string),
            "target_entity": target entity name (string),
            "target_type": target entity type (string),
            "description": explanation of why this relationship applies (string)
        }},
        ...
    ]
}}
</JSON_Response_Format>"""


class EntityRelationExtractor:
    def __init__(self, llm):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            ("user", USER_TEMPLATE),
        ])

        self.chain = prompt | llm | JsonOutputParser()
        self.entity_types = {json.dumps(GRAPH_SCHEMA["entity"], indent=2)}
        self.relationship_types = {json.dumps(GRAPH_SCHEMA["relationship"], indent=2)}
        
    def _validate_entity(self, entity):
        """엔티티의 유효성을 검증합니다."""
        if not isinstance(entity, dict):
            return False, "딕셔너리 형식이 아닙니다"
            
        if "type" not in entity or "name" not in entity:
            return False, "필수 필드(type, name)가 없습니다"
            
        if not isinstance(entity["type"], str) or not isinstance(entity["name"], str):
            return False, "type과 name은 문자열이어야 합니다"
            
        if entity["type"] not in ENTITY_TYPES:
            return False, f"잘못된 entity type입니다: {entity['type']}"
            
        return True, None
        
    def _validate_relationship(self, rel):
        """관계의 유효성을 검증합니다."""
        if not isinstance(rel, dict):
            return False, "딕셔너리 형식이 아닙니다"
            
        if "type" not in rel or "source_entity" not in rel or "source_type" not in rel or "target_entity" not in rel or "target_type" not in rel or "description" not in rel:
            return False, "필수 필드(type, source_entity, source_type, target_entity, target_type, description)가 없습니다"
            
        if not isinstance(rel["type"], str) or not isinstance(rel["source_entity"], str) or not isinstance(rel["source_type"], str) or not isinstance(rel["target_entity"], str) or not isinstance(rel["target_type"], str) or not isinstance(rel["description"], str):
            return False, "필드 타입이 잘못되었습니다"
            
        if rel["type"] not in RELATIONSHIP_TYPES:
            return False, f"잘못된 relationship type입니다: {rel['type']}"
                
        if rel["source_type"] not in ENTITY_TYPES:
            return False, f"잘못된 source_type입니다: {rel['source_type']}"
            
        if rel["target_type"] not in ENTITY_TYPES:
            return False, f"잘못된 target_type입니다: {rel['target_type']}"
            
        return True, None
        
    def _validate_response(self, response):
        """응답의 유효성을 검증합니다."""
        if not isinstance(response, dict):
            return False, "응답이 딕셔너리 형식이 아닙니다"
            
        if "entities" not in response:
            return False, "'entities' 필드가 없습니다"
            
        if not isinstance(response["entities"], list):
            return False, "'entities' 필드가 리스트 형식이 아닙니다"
            
        # entities 검증
        for entity in response["entities"]:
            is_valid, error = self._validate_entity(entity)
            if not is_valid:
                return False, f"잘못된 entity: {error}"
                
        # relationships 검증 (있는 경우)
        if "relationships" in response:
            if not isinstance(response["relationships"], list):
                return False, "'relationships' 필드가 리스트 형식이 아닙니다"
                
            for rel in response["relationships"]:
                is_valid, error = self._validate_relationship(rel)
                if not is_valid:
                    return False, f"잘못된 relationship: {error}"
                    
        return True, None

    def __call__(self, state):
        queries = [{"document": chunk.content, "entity_types": self.entity_types, "relationship_types": self.relationship_types} for chunk in state.chunks]
        with BatchCallback(len(queries)) as cb: # init callback
            responses = self.chain.batch(queries, config={"callbacks": [cb], "max_concurrency": 16})

        valid_chunks = []
        for i, (chunk, res) in enumerate(zip(state.chunks, responses)):
            is_valid, error = self._validate_response(res)
            if not is_valid:
                logger.warning(f"응답 #{i+1} 검증 실패")
                logger.warning(f"원인: {error}")
                logger.warning(f"응답: {res}")
                logger.warning("---")
            else:
                chunk.entities = res["entities"]
                chunk.relationships = res["relationships"]
                valid_chunks.append(chunk)

        # with open("./data/items.json", "w") as f:
        #     json.dump(valid_items, f)

        # with open("./data/responses.json", "w") as f:
        #     json.dump(valid_responses, f)

        # with open("./data/items.json", "r") as f:
        #     valid_items = json.load(f)

        # with open("./data/responses.json", "r") as f:
        #     valid_responses = json.load(f)
        
        return Command(
            update={
                "chunks": valid_chunks
            },
            goto="GraphDBWriter"
        )