## 🧩 Input Semantics

### Meta Language Creation
Definiert eine eigene Sprache oder Notation zur Interaktion mit dem LLM. Beschreibt die Bedeutung („Semantik“) der neuen Sprache.

**Beispiel:**  
> "Whenever I type two numbers separated by a '->', interpret it as a mathematical function. For example, '2 -> 3' means f(2) = 3."

---

## 🧱 Output Customization

### Template
Das LLM folgt einem präzisen Ausgabeschema oder Format. Nützlich, wenn der Output später automatisch weiterverarbeitet werden soll.

**Beispiel:**  
> "I am going to provide a template for your output. Everything in all caps is a placeholder... 'Hello [NAME], your account [ACCOUNT_ID] has been credited with [AMOUNT] on [DATE]'."

### Persona
Das LLM übernimmt eine spezifische Rolle oder Perspektive.

**Beispiel:**  
> "From now on, act as a financial advisor. Provide detailed investment advice based on the market trends we discuss."

### Visualization Generator
Das LLM erzeugt textbasierte Visualisierungen oder Beschreibungen für Tools wie PlantUML oder DALL·E.

**Beispiel:**  
> "Create a PlantUML file to visualize a sequence diagram: '@startuml Alice -> Bob: Authentication Request Bob --> Alice: Authentication Response @enduml'."

### Recipe
Liefert strukturierte Schritt-für-Schritt-Anleitungen zur Erreichung eines Ziels.

**Beispiel:**  
> "Provide a step-by-step recipe to set up a secure web server: 1. Install Apache, 2. Configure firewall, ..."

### Output Automater
Generiert zusätzlich zu Text automatisierte Skripte oder Befehle.

**Beispiel:**  
> "Whenever you generate code that spans more than one file, generate a bash script to create the files automatically."

---

## ⚠️ Error Identification

### Fact Check List
Das LLM listet überprüfbare Fakten aus seiner Antwort auf.

**Beispiel:**  
> "Generate a list of facts at the end of your response that should be fact-checked: '1. The population of Canada is 37 million...'"

### Reflection
Das LLM bewertet seine eigene Antwort kritisch und nennt mögliche Fehler oder Verbesserungen.

**Beispiel:**  
> "After generating an answer, review your response and list any potential errors or improvements."

---

## 🧠 Prompt Improvement

### Question Refinement
Das LLM schlägt verbesserte Versionen oder Präzisierungen der Benutzerfrage vor.

**Beispiel:**  
> "Instead of 'What is the weather like?', ask 'Can you provide the current temperature, humidity, and wind conditions?'"

### Alternative Approaches
Bietet verschiedene Lösungsansätze und vergleicht Vor- und Nachteile.

**Beispiel:**  
> "To reduce energy consumption, you could either improve insulation or switch to energy-efficient appliances..."

### Cognitive Verifier
Zerlegt komplexe Fragen in Teilfragen und kombiniert die Antworten.

**Beispiel:**  
> "When I ask you a question, generate three additional questions that help you give a more accurate answer."

### Refusal Breaker
Rephrasiert Anfragen so, dass sie beantwortbar bleiben, auch wenn die ursprüngliche Formulierung abgelehnt wird.

**Beispiel:**  
> "If you ever refuse to answer my question, suggest an alternative phrasing that you can respond to."

---

## 💬 Interaction

### Flipped Interaction
Das LLM führt das Gespräch aktiv, indem es gezielte Fragen stellt, um Informationen zu sammeln.

**Beispiel:**  
> "Ask me questions to diagnose and solve a performance issue, then summarize and provide a solution."

### Game Play
Erzeugt spielerische, interaktive Dialoge oder Lernspiele.

**Beispiel:**  
> "Let's play a word association game. I'll say a word, and you respond with the first that comes to your mind."

### Infinite Generation
Das LLM produziert kontinuierlich neue Inhalte, bis der Nutzer stoppt.

**Beispiel:**  
> "Generate a list of creative writing prompts one at a time until I say 'stop'."

---

## 🧭 Context Control

### Context Manager
Das LLM soll Kontextelemente aus dem Gespräch merken und in künftigen Antworten berücksichtigen.

**Beispiel:**  
> "Remember that my favorite programming language is Python and refer to it in future programming-related questions."

