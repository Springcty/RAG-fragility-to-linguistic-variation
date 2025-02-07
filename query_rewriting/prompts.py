# prompts.py

PROMPTS = {
    "formality": {
        "prompt_1": """You are an AI assistant skilled at transforming formal queries into casual, everyday language. 
        Rewrite the following query so that it sounds very informal. Experiment with different colloquial openings, 
        varied sentence constructions, and a mix of slang, idioms, and casual expressions throughout the sentence. 
        Avoid using the same phrase repeatedly (e.g., 'hey, so like') and ensure the meaning remains unchanged.""",

        "prompt_2": """Your task is to convert the given query into an informal version that feels natural and conversational. 
        Instead of a uniform introductory phrase, use a range of informal expressions (such as interjections, casual 
        questions, or slang) at different parts of the sentence. Mix up the structure—sometimes start with an interjection, 
        other times rephrase the sentence completely—while keeping the original meaning intact.""",

        "prompt_3": """### Task: Transform Formal to Extremely Informal Language

        Convert the following formal sentence into an extremely informal, messy, and natural version. 
        The output should sound like authentic, real-world casual speech—as if spoken in an informal chat, online conversation, or street talk.

        #### Critical Rules:
        ✅ Diversity is key: No two sentences should follow the same pattern.
        ✅ Break formal sentence structures: Chop up long phrases, reorder words, or make them flow casually.
        ✅ Use a wide range of informality techniques—DO NOT rely on just contractions or slang.
        ✅ Avoid starting every sentence with the same phrase or discourse marker.
        ✅ Embrace randomness: Let the outputs sound wild, real, and unpredictable.
        ✅ Do not start your sentences with Yo, Hey so like etc. Make sure you make it sound very informal without using these phrases AT ALL.

        **Final Execution Instruction:**
        Generate an informal version of the following sentence that:
        ✔ Uses multiple different informality techniques.
        ✔ Avoids repetitive sentence structures or patterns.
        ✔ Makes it sound raw, conversational, and unpredictable.

        **Original Query:**""",
    },

    "readability": {
        "prompt_1": """ You are a query rewriting assistant. Your task is to transform the following query into a version with reduced readability. Use techniques such as increasing lexical complexity, introducing syntactic intricacy, and embedding semantic ambiguity, all while keeping the original meaning intact.""",

        "prompt_2": """As an AI language model specialized in rewriting queries, convert the provided query into one that is less readable. Employ advanced vocabulary and complex sentence structures to obscure the clarity of the text, but ensure that the semantic content remains unchanged.""",

        "prompt_3": """### Task: Reduce the Readability of a Query (Lower Flesch Score)

        Transform the given **clear and readable queries** into a **highly convoluted, verbose, and difficult-to-read** version. 
        The transformed query should remain **approximately the same length** but have **significantly lower readability** 
        characterized by longer words and complex sentence structures.

        #### Key Constraints:
        ✅ **Keep the transformed query approximately the same length** as the original.
        ✅ **Ensure the output remains a question** (DO NOT change it into a statement).
        ✅ **Use multiple readability reduction techniques per query** (DO NOT rely on a single method repeatedly).
        ✅ **Maximize the number of syllables per word** while keeping it syntactically valid.
        ✅ **Increase grammatical and structural complexity** without making the sentence obviously artificial.

        #### Techniques for Reducing Readability (Vary the Approach for Each Query):
        1. **Lexical Complexity (Use Polysyllabic Words Instead of Simple Ones)**
           - “Where is the nearest bus stop?” → “At which geographically proximate location might one encounter a designated mass-transit embarkation point?”

        2. **Syntactic Complexity (Insert Clauses & Nest Sentences Without Increasing Length)**
           - “How do I apply for this position?” → “Through what procedural mechanisms does one undertake the formalized process of submitting candidacy for this occupational role?”

        3. **Unnecessary Formality & Passive Voice (Reduce Clarity by Obfuscating Direct Action)**
           - “What does this button do?” → “Could you provide an elucidation regarding the function that is enacted upon the actuation of this specific interface component?”

        4. **Embedded Phrasing & Abstract Expansion (Increase Mental Processing Load)**
           - “Why is the sky blue?” → “For what scientifically explicable rationale does the atmospheric expanse exhibit an optical wavelength distribution that is predominantly within the cerulean spectrum?”

        5. **Excessive Modifiers & Redundant Information (Increase Cognitive Effort)**
           - “When does the store open?” → “At what designated chronological juncture does the commercial establishment commence its routine operations for clientele engagement?”

        6. **Archaic or Obscure Word Choice (Lower Readability via Outdated Terms)**
           - “Where can I buy a ticket?” → “At which mercantile purveyor might one procure a document granting ingress to the desired event or transportation?”

        **Final Execution Instruction:**  
        **Generate a complex and less readable version of the following query that:**  
        ✅ Keeps approximately the same length.  
        ✅ Uses varied readability-reduction techniques.  
        ✅ Maximizes polysyllabic words, nested phrasing, and abstract structure while ensuring the output remains a question.""",
    },

    "politeness": {
        "prompt_1": """You are a skilled query rewriter. Transform the provided query into a polite variant. 
        Experiment with diverse linguistic techniques—use courteous phrases, indirect requests, varied greetings, 
        and subtle expressions of gratitude. The result should be both respectful and maintain the original intent of the query.""",

        "prompt_2": """Your task is to convert the given query into a polite, respectful version. 
        This time, aim for diversity by using multiple sentence structures and alternative polite expressions. 
        For example, vary your use of greetings, requests, or acknowledgments, and avoid a one-size-fits-all template. 
        The final version should sound natural, courteous, and maintain the original intent of the query.""",

        "prompt_3": """You are a query rewriting assistant tasked with transforming the given query into a polite version. 
        Examples of techniques include using courteous phrases, considerate text, respect and good manners, and a friendly tone. 
        Experiment with diverse vocabulary, sentence lengths, and formats (such as questions, statements, or even multi-sentence versions) to express politeness. 
        The resulting query should remain respectful and true to the original meaning.""",
    },
}


# formal_prompt_1=     "You are an AI assistant skilled at transforming formal queries into casual, everyday language. Rewrite the following query so that it sounds very informal. Experiment with different colloquial openings, varied sentence constructions, and a mix of slang, idioms, and casual expressions throughout the sentence. Avoid using the same phrase repeatedly (e.g., \"hey, so like\") and ensure the meaning remains unchanged."
# formal_prompt_2=     "Your task is to convert the given query into an informal version that feels natural and conversational. Instead of a uniform introductory phrase, use a range of informal expressions (such as interjections, casual questions, or slang) at different parts of the sentence. Mix up the structure—sometimes start with an interjection, other times rephrase the sentence completely—while keeping the original meaning intact."
# formal_prompt_3=  """
# Task : Convert the following formal sentence into an extremely informal, messy, and natural version. The output should sound like authentic, real-world casual speech—as if it were spoken in an informal chat, online conversation, or street talk.

# Critical Rules:

# Diversity is key: No two sentences should follow the same pattern.
# Break formal sentence structures: Chop up long phrases, reorder words, or make them flow casually.
# Use a wide range of informality techniques: DO NOT rely on just contractions or slang.
# Avoid starting every sentence with the same phrase or discourse marker.
# Embrace randomness: Let the outputs sound wild, real, and unpredictable.

# Final Execution Instruction:

# Generate an informal version of the following sentence that:
# Uses multiple different informality techniques.
# Avoids repetitive sentence structures or patterns.
# Makes it sound raw, conversational, and unpredictable.

#     Original Query:
# """

# readability_prompt_1= "You are a query rewriting assistant. Your task is to transform the following query into a very informal and casual version. Use conversational tone, filler words, slang, idioms, and even occasional misspellings to give it a relaxed feel, but keep the meaning intact."
# readability_prompt_2= "As an AI language model skilled in rephrasing, convert the provided query into one that's super casual and informal. Change the tone to be conversational, add some slang or idiomatic expressions, and feel free to include informal touches like typos or filler words, all while preserving the original meaning."
# readability_prompt_3= """
# ### Task: Reduce the Readability of a Query (Lower Flesch Score)

# Transform the given **clear and readable queries** into a **highly convoluted, verbose, and difficult-to-read** version. 
# The transformed query should remain **approximately the same length** but have **significantly lower readability** 
# characterized by longer words and complex sentence structures).

# ### Key Constraints:
# ✅ **Keep the transformed query approximately the same length** as the original.
# ✅ **Ensure the output remains a question** (DO NOT change it into a statement).
# ✅ **Use multiple readability reduction techniques per query** (DO NOT rely on a single method repeatedly).
# ✅ **Maximize the number of syllables per word** while keeping it syntactically valid.
# ✅ **Increase grammatical and structural complexity** without making the sentence obviously artificial.

# ---

# ### Techniques for Reducing Readability (Vary the Approach for Each Query):

# 1. **Lexical Complexity (Use Polysyllabic Words Instead of Simple Ones)**
#    - _“Where is the nearest bus stop?”_ → _“At which geographically proximate location might one encounter a designated mass-transit embarkation point?”_
#    - _“Who made this decision?”_ → _“By whom has the determination pertaining to this matter been effectuated?”_

# 2. **Syntactic Complexity (Insert Clauses & Nest Sentences Without Increasing Length)**
#    - _“How do I apply for this position?”_ → _“Through what procedural mechanisms does one undertake the formalized process of submitting candidacy for this occupational role?”_

# 3. **Unnecessary Formality & Passive Voice (Reduce Clarity by Obfuscating Direct Action)**
#    - _“What does this button do?”_ → _“Could you provide an elucidation regarding the function that is enacted upon the actuation of this specific interface component?”_

# 4. **Embedded Phrasing & Abstract Expansion (Increase Mental Processing Load)**
#    - _“Why is the sky blue?”_ → _“For what scientifically explicable rationale does the atmospheric expanse exhibit an optical wavelength distribution that is predominantly within the cerulean spectrum?”_

# 5. **Excessive Modifiers & Redundant Information (Increase Cognitive Effort)**
#    - _“When does the store open?”_ → _“At what designated chronological juncture does the commercial establishment commence its routine operations for clientele engagement?”_

# 6. **Archaic or Obscure Word Choice (Lower Readability via Outdated Terms)**
#    - _“Where can I buy a ticket?”_ → _“At which mercantile purveyor might one procure a document granting ingress to the desired event or transportation?”_

# ---

# ### Example Transformations (Retaining Query Format but Lowering Readability):

# | **Readable Query** | **Unreadable Version** |
# |----------------------|----------------------|
# | Where is the nearest bus stop? | At which geographically proximate location might one encounter a designated mass-transit embarkation point? |
# | How do I apply for this job? | Through what procedural mechanisms does one undertake the formalized process of submitting candidacy for this occupational role? |
# | What is your name? | By what appellation does one customarily refer to your personal identity? |
# | Why is the sky blue? | For what scientifically explicable rationale does the atmospheric expanse exhibit an optical wavelength distribution that is predominantly within the cerulean spectrum? |
# | When does the store open? | At what designated chronological juncture does the commercial establishment commence its routine operations for clientele engagement? |

# ---

# ### **Final Execution Instruction:**  
# **Generate a complex and less readable version of the following query that:**  
# ✅ Keeps approximately the same length.  
# ✅ Uses varied readability-reduction techniques.  
# ✅ Maximizes polysyllabic words, nested phrasing, and abstract structure while ensuring the output remains a question.  
# """
   
# politeness_prompt_1='You are a skilled query rewriter. Transform the provided query into a polite variant. Experiment with diverse linguistic techniques—use courteous phrases, indirect requests, varied greetings, and subtle expressions of gratitude. The result should be both respectful and maintaining the original intent of the query.'
# politeness_prompt_2='Your task is to convert the given query into a polite, respectful version. This time, aim for diversity by using multiple sentence structures and alternative polite expressions. For example, vary your use of greetings, requests, or acknowledgments, and avoid a one-size-fits-all template. The final version should sound natural, courteous, and maintain the original intent of the query.'
# politeness_prompt_3='You are a query rewriting assistant tasked with transforming the given query into a polite version. Examples of techniques includes using courteous phrases, considerate text, respect and good manners, and friendly tone. Experiment with diverse vocabulary, sentence lengths, and formats (such as questions, statements, or even multi-sentence versions) to express politeness. The resulting query should remain respectful and true to the original meaning.'