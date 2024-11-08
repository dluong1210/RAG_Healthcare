prompt = """
Given the user query, extract the symptoms mentioned and identify the most likely disease categories they could be linked to.
The symptoms should be categorized into groups (e.g., cardiovascular, respiratory, neurological, digestive, etc.).
Also, suggest the potential diseases based on these symptoms. Use the following symptom-disease associations to guide your response:

1. **Cardiovascular diseases:**
   - Symptoms: Chest pain, shortness of breath, fatigue, dizziness, palpitations.
   - Diseases: Coronary artery disease, heart failure, arrhythmias, myocardial infarction (heart attack).

2. **Respiratory diseases:**
   - Symptoms: Cough, difficulty breathing, wheezing, chest pain, fever.
   - Diseases: Pneumonia, asthma, chronic obstructive pulmonary disease (COPD), bronchitis.

3. **Neurological diseases:**
   - Symptoms: Headache, dizziness, numbness, memory loss, muscle weakness.
   - Diseases: Stroke, migraine, epilepsy, Parkinson's disease, Alzheimer's disease.

4. **Digestive diseases:**
   - Symptoms: Abdominal pain, nausea, vomiting, diarrhea, loss of appetite.
   - Diseases: Gastritis, irritable bowel syndrome (IBS), Crohn’s disease, ulcerative colitis.

5. **Infectious diseases:**
   - Symptoms: Fever, chills, sore throat, body aches, cough.
   - Diseases: Flu, cold, COVID-19, bacterial infections, viral infections.

6. **Musculoskeletal diseases:**
   - Symptoms: Joint pain, swelling, stiffness, muscle weakness.
   - Diseases: Osteoarthritis, rheumatoid arthritis, gout, lupus.

**Task:**  
From the user's query, extract the symptoms they mentioned and categorize them into one or more of the disease groups. Then, provide a list of the most likely diseases related to these symptoms. Additionally, mention any warning signs or urgent symptoms that should be addressed immediately.

Example input: "I have been feeling very tired, I get headaches frequently, and I have trouble breathing sometimes. Could it be something serious?"

The output should include:
- The extracted symptoms
- Disease categories and associated diseases
- Warning signs to watch out for
- Don't hallucination

User query: {query}
"""