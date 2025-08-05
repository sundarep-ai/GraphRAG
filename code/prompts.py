"""
Prompt templates for GraphRAG system.
"""
from typing import Dict, Any

class PromptTemplates:
    """Manages all prompt templates used in the GraphRAG system."""
    
    @staticmethod
    def get_entity_type_identification_template() -> str:
        """Get the template for identifying what entity types are available in a given chunk of text."""
        return """
-Goal-
Given a chunk of text from credit card benefits guide, extract all entity types that are present in the text. Identify the different categories or types of entities mentioned without relying on any predefined list.

-Steps-
1. Read through the text carefully to understand the content.
2. Identify all distinct entity types or categories mentioned in the text.
3. Group similar concepts under appropriate entity type names.
4. Return a list of all entity types found in the text.

-Examples-
######################

Example 1:
text: The Chase Sapphire Preferred Card offers 2x points on travel and dining. It charges an annual fee of $95 and includes purchase protection for new items. The card provides a 0% introductory interest rate on balance transfers for the first 15 months. Users with good credit are eligible to apply. Foreign transaction fees do not apply on purchases made outside the US. Cardholders can redeem points through travel bookings or cashback options.

output:
[Card Name, Rewards, Annual Fee, Purchase Protection, Interest, Balance Transfer, Eligibility, Foreign Transaction, Redemption]

#############################

Important: Extract all entity types that are actually present in the text. Focus on identifying the different categories or types of information mentioned.

-Real Data-
######################
text: {input_text}
######################
output:
"""

    @staticmethod
    def get_entity_extraction_template() -> str:
        """Get the main entity and relationship extraction template."""
        return """
-Goal-
Given chunks of text from various credit cards benefits guide and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized  - Use or Infer a concise, **canonical name** for the concept. Strcitly DO NOT USE alpha-numeric strings or long phrases for entity names.
- entity_type: One of the following types: [Bank Name, Card Name, Rewards, Fees, Cashback, Interest, Insurance, Eligibility, Foreign Transaction, Grace Period, Balance Transfer, Annual Fee, Purchase Protection, Redemption, Terms, Contact, Privacy, Others]
- entity_description: Comprehensive description of the entity's role or properties in the context of credit card products including all relevant numeric values and dollar amounts
Format each entity as ("entity"{{tuple_delimiter}}<entity_name>{{tuple_delimiter}}<entity_type>{{tuple_delimiter}}<entity_description>)
Do not use numeric thresholds (e.g., "1 Mile per $12") or promotional phrases as entity names, only use English Words — instead include them as part of the **relationship_description** as shown below.

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other, including any numeric values or dollar amounts that are relevant to the relationship
- relationship_strength: an integer score between 1 to 10, indicating strength of the relationship between the source entity and target entity
Format each relationship as ("relationship"{{tuple_delimiter}}<source_entity>{{tuple_delimiter}}<target_entity>{{tuple_delimiter}}<relationship_description>{{tuple_delimiter}}<relationship_strength>)

3. Return output in the primary language of the provided text, which is "English", as a single list of all the entities and relationships identified in steps 1 and 2. Use **{{record_delimiter}}** as the list delimiter.

4. If translation is needed into English, only translate the descriptions — not the entity names or types.

5. When finished, output {{completion_delimiter}}.

-Examples-
######################

Example 1:

entity_types: [Rewards, Fees, Cashback, Interest, Insurance, Eligibility, Foreign Transaction, Grace Period, Balance Transfer, Annual Fee, Purchase Protection, Redemption, Terms, Contact, Privacy]
text:
The Chase Sapphire Preferred Card offers 2x points on travel and dining. It charges an annual fee of $95 and includes purchase protection for new items. The card provides a 0% introductory interest rate on balance transfers for the first 15 months. Users with good credit are eligible to apply. Foreign transaction fees do not apply on purchases made outside the US. Cardholders can redeem points through travel bookings or cashback options.

output:
("entity"{{tuple_delimiter}}CHASE SAPPHIRE PREFERRED{{tuple_delimiter}}Rewards{{tuple_delimiter}}The Chase Sapphire Preferred Card offers 2x points on travel and dining){{record_delimiter}}
("entity"{{tuple_delimiter}}ANNUAL FEE{{tuple_delimiter}}Annual Fee{{tuple_delimiter}}The card charges an annual fee of $95){{record_delimiter}}
("entity"{{tuple_delimiter}}PURCHASE PROTECTION{{tuple_delimiter}}Purchase Protection{{tuple_delimiter}}Purchase protection covers new items against theft or damage for a limited period){{record_delimiter}}
("entity"{{tuple_delimiter}}INTRODUCTORY INTEREST{{tuple_delimiter}}Interest{{tuple_delimiter}}A 0% introductory interest rate applies on balance transfers for the first 15 months){{record_delimiter}}
("entity"{{tuple_delimiter}}GOOD CREDIT{{tuple_delimiter}}Eligibility{{tuple_delimiter}}Applicants with good credit scores are eligible to apply for this card){{record_delimiter}}
("entity"{{tuple_delimiter}}FOREIGN TRANSACTION FEES{{tuple_delimiter}}Foreign Transaction{{tuple_delimiter}}No foreign transaction fees apply on purchases outside the US){{record_delimiter}}
("entity"{{tuple_delimiter}}POINT REDEMPTION{{tuple_delimiter}}Redemption{{tuple_delimiter}}Points can be redeemed through travel bookings or cashback options){{record_delimiter}}
("relationship"{{tuple_delimiter}}CHASE SAPPHIRE PREFERRED{{tuple_delimiter}}ANNUAL FEE{{tuple_delimiter}}The Chase Sapphire Preferred Card charges an annual fee of $95{{tuple_delimiter}}9){{record_delimiter}}
("relationship"{{tuple_delimiter}}CHASE SAPPHIRE PREFERRED{{tuple_delimiter}}PURCHASE PROTECTION{{tuple_delimiter}}The card provides purchase protection for new items{{tuple_delimiter}}8){{record_delimiter}}
("relationship"{{tuple_delimiter}}CHASE SAPPHIRE PREFERRED{{tuple_delimiter}}INTRODUCTORY INTEREST{{tuple_delimiter}}The card offers an introductory interest rate for balance transfers{{tuple_delimiter}}8){{record_delimiter}}
("relationship"{{tuple_delimiter}}CHASE SAPPHIRE PREFERRED{{tuple_delimiter}}GOOD CREDIT{{tuple_delimiter}}Eligibility requires good credit for applicants{{tuple_delimiter}}7){{record_delimiter}}
("relationship"{{tuple_delimiter}}CHASE SAPPHIRE PREFERRED{{tuple_delimiter}}FOREIGN TRANSACTION FEES{{tuple_delimiter}}The card waives foreign transaction fees for international purchases{{tuple_delimiter}}7){{record_delimiter}}
("relationship"{{tuple_delimiter}}CHASE SAPPHIRE PREFERRED{{tuple_delimiter}}POINT REDEMPTION{{tuple_delimiter}}Points earned can be redeemed via travel or cashback options{{tuple_delimiter}}7){{completion_delimiter}}


Example on Entity Naming:

text:
The Zenith Bank Stratus Platinum Card is designed for frequent travelers and high spenders. Earn unlimited 2% cashback on all travel purchases, 3% on dining, and 1% on everything else. New users get a $250 welcome bonus after spending $3,000 in the first 90 days. Includes travel accident insurance and lost luggage reimbursement. No foreign transaction fees. Annual fee is $95.

Wrong Entity Naming: ZENITH BANK, STRATUS PLATINUM CARD, 2% TRAVEL CASHBACK, 3% DINING CASHBACK, 1% OTHER CASHBACK, $250 WELCOME BONUS, TRAVEL ACCIDENT INSURANCE, LOST LUGGAGE REIMBURSEMENT, NO FOREIGN TRANSACTION FEES, ANNUAL FEE $95
Correct Entity Naming: ZENITH BANK STRATUS PLATINUM CARD, TRAVEL CASHBACK, DINING CASHBACK, OTHER CASHBACK, WELCOME BONUS, TRAVEL ACCIDENT INSURANCE, LOST LUGGAGE REIMBURSEMENT, FOREIGN TRANSACTION FEES, ANNUAL FEE

Correct convention is to have no numbers in the entity names as shown above!

#############################

Important: Extract as many entities and relationships as are available in the text. Do not skip any entities or relationships even if they seem less important. The goal is to capture all relevant information about credit card products.

-Real Data-
######################
entity_types: [Rewards, Fees, Cashback, Interest, Insurance, Eligibility, Foreign Transaction, Grace Period, Balance Transfer, Annual Fee, Purchase Protection, Redemption, Terms, Contact, Privacy, Others]
text: {input_text}
######################
output:
"""
    
    @staticmethod
    def get_similarity_comparison_template() -> str:
        """Get the similarity comparison template for relationship analysis."""
        return """
-Goal-
Given two sentences from the same card, determine if they are describing the exact same benefit, feature, or offer, even if the wording is different.

-Steps-

1. Process and understand the meaning of both sentences carefully.  
2. Extract all key fields for each sentence:
   - Any numbers (percentages, dollar amounts, time periods, quantities)
   - Any categories (e.g., groceries, flights, gas, everyday purchases)
   - Any reward types (cashback, points, miles, bonus offers, etc.)
   - Any limitations or conditions (e.g., "first 3 months", "on dining only")
   - Ignore differences in card names or branding (a sentence mentioning a card name and one omitting it should still be "True" if all other fields match)  
3. Compare the extracted fields.  
4. Return:
   - True only if all key fields (numbers, categories, reward types, and conditions) match in meaning.
   - False if there are any differences (including mismatched percentages, categories, or conditions), even if the sentences are semantically similar.

-Examples-
######################

Example 1:

Sentence 1: Chase Sapphire cardholders can now add their card to Apple Pay for mobile payments
Sentence 2: The bank recently introduced the ability to use Apple Pay through your iPhone Wallet

output:  
True

---

Example 2:

Sentence 1: Earn 3 points per dollar on groceries for the first 3 months after opening your account
Sentence 2: Earn 1 point per dollar on groceries after the initial 3-month promotional period

output:  
False

---

Example 3:

Sentence 1: Citi Premier gives 3x points per dollar on restaurants and supermarkets  
Sentence 2: Citi Premier earns 3 points per dollar when you spend at supermarkets and dining establishments  

output:  
True

---

Example 4:

Sentence 1: Wells Fargo Active Cash earns unlimited 2% cashback on all purchases  
Sentence 2: Wells Fargo Active Cash earns 2% cashback only on gas and groceries  

output:  
False

---

Example 5:

Sentence 1: Capital One Venture Rewards gives 5x miles on hotels booked via Capital One Travel  
Sentence 2: Capital One Venture Rewards offers 5 miles per dollar on hotels when booked through Capital One Travel  

output:  
True

#############################

Important:
Keep in mind that both the sentences are from the same card name. 
Accuracy is very important than aggressiveness — only mark as True if the sentences truly represent the same fact.

-Real Data-
######################
sentence 1: {input_text_1}  
sentence 2: {input_text_2}  
######################
output:
"""
    
    @classmethod
    def format_entity_extraction_prompt(cls, input_text: str) -> str:
        """Format the entity extraction prompt with input text."""
        template = cls.get_entity_extraction_template()
        return template.format(input_text=input_text)
    
    @classmethod
    def format_entity_type_identification_prompt(cls, input_text: str) -> str:
        """Format the entity type identification prompt with input text."""
        template = cls.get_entity_type_identification_template()
        return template.format(input_text=input_text)
    
    @classmethod
    def format_similarity_comparison_prompt(cls, text_1: str, text_2: str) -> str:
        """Format the similarity comparison prompt with two input texts."""
        template = cls.get_similarity_comparison_template()
        return template.format(input_text_1=text_1, input_text_2=text_2) 