## Topic5rag


### Teammates

- Albert Huang
- Jacob Huynh


## Assignment Answers

### Exercise 1

#### Model T manual queries
- **No RAG:** Frequently invents procedures and specs. Specs vary across runs -> low reliability.
- **With RAG:** Often pulls manual-adjacent instructions (e.g., band adjustment screw/lock nut; carburetor rod install). Still inconsistent on numeric specs (spark plug gap differs across indexes). Oil question not answered well (mostly “add new oil” / “approved oils”).

#### Congressional Record queries
- **No RAG:** Makes up detailed claims about specific people/events.
- **With RAG:** Usually more grounded for “who said what” and bill purpose.

#### Mixed (Model T + CR in one index)
- Benign when retrieval lands in the right corpus, but increases risk of wrong-corpus / wrong-passage hits.

### Exercise 2

> Open Model + RAG comparison reuses results from exercise 1.

**Comparison: Qwen 2.5 1.5B (with RAG) vs. GPT-4o Mini (No Tools)**

* **Hallucination Control:** GPT-4o Mini is significantly better at avoiding "low-level" hallucinations (like making up mechanical steps for a carburetor it doesn't know) but fails fundamentally on temporal data. It admitted ignorance for the 2026 Congressional Record queries, whereas Qwen 2.5 1.5B without RAG confidently invented a fake 2026 hearing for Mr. Flood.
* **Performance on Corpora:**
    * **Model T Ford:** GPT-4o Mini appeared to have performed well here. Because the Model T was designed in 1908, it is heavily represented in the pre-training data. It provided spark plug gaps (0.025") and oil recommendations without RAG.
    * **Congressional Record (Jan 2026):** GPT-4o Mini failed entirely, citing its October 2023 knowledge cutoff. In contrast, the small Qwen model with RAG was able to correctly identify Mr. Flood's remarks because the information was provided in the context window.
* **Conclusion:** A "Large" model is only better if the information is historical. For private or future-dated data (like your 2026 corpora), a small model with RAG is strictly superior to a frontier model without it.

### Exercise 3: Open Model + RAG vs. State-of-the-Art Chat Model

Setup:
Using Claude Sonnet 4.6 with extended thinking and web search enabled

Instructions specify using the RAG responses for the Model T questions so I will only feed Claude Sonnet 4.6 the Model T questions and not those about the Congressional Records.

- "How do I adjust the carburetor on a Model T?"

Answer:
The Model T Ford used a simple Kingston or Holley carburetor (depending on the year), and adjusting it is pretty straightforward. Here's how:

**Basic Carburetor Adjustment**

The main adjustment you'll make is the **needle valve (mixture screw)**, which controls the fuel-air ratio at idle and low speeds.

1. **Warm up the engine first.** Start the car and let it run until it reaches operating temperature. Use the hand throttle lever on the steering column to set a fast idle.

2. **Locate the mixture needle.** On the Holley vaporizer (used on most 1915–1927 models), there's a needle valve on the side of the carburetor body. On Kingston carbs (earlier models), it's similarly accessible.

3. **Adjust the needle.** Turn it clockwise (in) to lean the mixture, counterclockwise (out) to richen it. Start by turning it gently in until the engine begins to stumble, then back it out slowly until the engine runs smoothly and picks up speed well. Typically you'll end up somewhere between 1 and 2.5 turns out from lightly seated.

4. **Set the idle speed.** Use the hand throttle lever on the steering column — there's no separate idle screw like on modern carbs. The spark lever (left lever) also affects smoothness; retarding it slightly can help at idle.

5. **Test under load.** With the mixture set, test acceleration. If it stumbles or "flat spots" when you open the throttle, richen the mixture slightly. If it smokes or runs rough at cruise, lean it a bit.

**A few Model T-specific tips:**

- The Holley vaporizer is sensitive to fuel level in the bowl — make sure your gravity-feed tank isn't running low, as this mimics a lean condition.
- The Model T runs on a very low compression ratio, so it's fairly forgiving of mixture settings.
- If the car runs better with the choke partially closed even when warm, the mixture is too lean.
- Always adjust with the spark lever in the normal running position (about halfway advanced).

If you're working on a specific year or carburetor brand and running into a particular issue (flooding, hard starting, stumbling), happy to help troubleshoot further.

- "What is the correct spark plug gap for a Model T Ford?"

Answer:
The correct spark plug gap for a Model T Ford is **0.035 inches** (approximately 0.9 mm). This was the standard factory specification, though some sources suggest a range of 0.030–0.040 inches was acceptable depending on the ignition system and engine condition.

- "How do I fix a slipping transmission band?"

Answer:
Fixing a slipping transmission band depends on whether you're dealing with an automatic or manual transmission, but here's a practical overview:

#### Diagnosing the Problem First

Before diving in, confirm it's actually the band. Symptoms include delayed shifting, RPMs revving high without corresponding acceleration, or the vehicle slipping between gears. Check the transmission fluid level and condition first — low or burnt fluid can mimic band slippage and is a much cheaper fix.

#### Adjustment (Automatic Transmission)

Many automatic transmissions have adjustable bands, and a simple adjustment can fix slipping without a full rebuild:

1. **Locate the adjustment screw** — it's usually on the side of the transmission case. Consult your vehicle's service manual for the exact location (it varies by make/model).
2. **Loosen the locknut** (typically 3/4" or similar) around the adjustment screw.
3. **Torque the adjustment screw** to the spec in your service manual — commonly around 72 in-lbs (6 ft-lbs), then back it off a specific number of turns (e.g., 2 turns for a Chrysler A727, 3.5 turns for some Ford units).
4. **Hold the screw in place** while retightening the locknut.

This is a relatively accessible DIY job with basic hand tools and the right service manual.

#### Replacement (If Adjustment Doesn't Help)

If the band material is worn through or the band is broken, it needs to be replaced. This involves:

- Dropping the transmission pan and valve body
- Sometimes removing the transmission entirely depending on the design
- Accessing the servo piston and band assembly
- Installing a new band and properly adjusting it

This is significantly more involved and typically requires transmission-specific knowledge and tools. Most people take this route to a shop.

#### Key Tips

- **Always use a service manual** specific to your transmission (ATSG manuals are great for automatics). Band adjustment specs vary widely between transmissions.
- **Check the fluid first** — do a drain and refill with the correct fluid type before anything else.
- **A transmission additive** like Lucas Transmission Fix sometimes temporarily resolves mild slipping from worn seals/bands, buying time.
- If the transmission has high mileage and multiple symptoms, a full rebuild may be more cost-effective long-term than chasing individual band issues.

What vehicle/transmission are you working with? I can give more specific guidance if you have the make and model.

- "What oil should I use in a Model T engine?"

Answer:
The Model T predates modern oil classifications, so you need to think about it differently than a modern engine.

#### What Ford Originally Specified

Henry Ford recommended a relatively light oil — roughly equivalent to a modern **SAE 20 or SAE 30** straight-weight (non-detergent) mineral oil. In winter, Ford suggested even lighter oil to aid cold starting.

#### What Most Model T Owners Use Today

The general consensus among the Model T Ford Club of America and experienced restorers is:

**Non-detergent SAE 30** is the most commonly recommended oil for regular use. The key word is _non-detergent_ — modern detergent oils can dislodge decades of accumulated deposits in old engines, causing those loosened particles to clog the simple splash lubrication system the T uses. If the engine has been recently rebuilt with new seals, detergent oil is slightly less of a concern, but non-detergent is still preferred.

In colder climates, **non-detergent SAE 20** is a reasonable winter choice since the T has no oil pump — it relies entirely on a splash/dipper system, and thicker oil won't circulate as well when cold.

#### A Few Practical Notes

- **Avoid synthetics and modern multi-viscosity oils** — they're generally overkill and the additives aren't always compatible with the old metallurgy, cork gaskets, and babbit bearings.
- The T's **babbit bearings** are softer than modern steel bearings and actually do well with straight mineral oil.
- **Change it frequently** — the T has minimal filtration, so short oil change intervals (every 500–1000 miles) are wise.
- Some restorers use oils marketed specifically for vintage/antique engines, like **Kendall GT-1 non-detergent** or **Castrol Classic** oils, which are formulated with older engine designs in mind.

The simplicity of the engine is actually forgiving — the main thing is avoiding modern detergent oils and keeping the oil fresh.

Discussion:

While the jump in improvement of response quality from large to frontier model is smaller than that of a small open to large model, it is still noticeably better. Responses for the most part are similar to that of GPT-4o-Mini but in the question about the carburetor, Sonnet 4.6 is able to identify the possible models of said carburetor and even give Model T specific tips on how to take care of the car. In the question for the spark plug gap, Sonnet 4.6 gives a definitive confident answer rather than a range from GPT-4o-Mini or no answer from our Qwen-2.5-1.5B + RAG setup. Finally, the response for the question about the engine oil has a section about "What Ford Originally Specified" along with what Model T owners in the present day use, providing a robust answer that is the best of both worlds. Interestingly, the frontier model did not use live web search on any of these questions, supporting the idea that model already had intrinsic knowledge of all of these concepts and could give confident answers. Frontier models only need to use live web search if it is on a very recent or niche topic that it is unsure about. The Model T, however, was much in the past and is a relevant popular topic that is still talked about to this day. Similar to the previous exercise with GPT-4o-Mini, the Qwen-2.5-1.5B + RAG setup still won on the question about changing transmission bands since it was able to give a really detailed targeted explanation pertaining specifically to the Model T with references to the paragraphs and figures. This supports the idea that RAG is similar to a live web search but on a knowledge base, adding value to a model when it is unsure of a topic that is maybe too recent or very niche and specific. By giving the model access to an external "database", it is able to fetch relevant information as context that it can directly refer to (similar to us using Google). Something intuitive but still interesting is that in the 3rd question that asks about fixing a slipped transmission band, a lack of mentioning of Model T meant that the model gave a modern answer referencing the cars we have today. This was attributed to every question being treated as an individual query instead of a conversation like in previous parts.

### Exercise 4: Effect of Top-K Retrieval Count

I am using the Congressional Records corpus and questions (provided) since the answers can be compared and are verifiable.

To quantitatively measure reponse latency, I adjusted the code to also print out response generation time.

Answer lengths are capped at 500 per the notebook's implementation so that responses aren't too long.

============================================================
TOP_K = 1
============================================================
Mr. Flood had to say that Mayor David Black is a paragon of public service and remarkable stewardship. He has been serving the people of Papillion for over two decades.

Relevant quote from the context: "The SPEAKER pro tempore. The Chair now recognizes the gentleman from Nebraska, Mr. FLOOD, for 5 minutes." followed by "Mr. FLOOD. Mr. Speaker, I rise today to recognize Papillion, Nebraska's, Mayor David Black, as a paragon of public service and remarkable stewardship."

This directly states th...

[Latency: 7.81s]

============================================================
TOP_K = 3
============================================================
Mr. Flood recognized Mayor David Black as a "paragon of public service" and a "remarkable steward," highlighting his contributions over two decades of service to the city of Papillion. He noted that despite his decision not to run for re-election, Mayor Black would continue to serve the city through various roles. The recognition was part of the Congressional Black Caucus's efforts to carry forward the legacy of Congressman John Lewis.

[Latency: 3.47s]

============================================================
TOP_K = 5
============================================================
Mr. Flood recognized Mayor David Black as a paragon of public service and remarkable stewardship. He highlighted Black's contributions over nearly 171/2 years of service, emphasizing his leadership role in guiding Papillion into a thriving economic hub. The speaker commended Black's dedication and thanked him for his outstanding public service.

[Latency: 2.85s]

============================================================
TOP_K = 10
============================================================
In the context provided, Mr. Flood recognized Mayor David Black of Papillion, Nebraska, during a speech on January 13, 2026. He praised Mayor Black's contributions to the city over nearly 5 years of service, highlighting his leadership in guiding the city into a thriving economic hub through partnerships with professionals and the business community. The recognition emphasized Black's dedication to public service and his role in building a strong foundation for future generations. Additionally, ...

[Latency: 8.35s]

============================================================
TOP_K = 1
============================================================
According to the context, Elise Stefanik made several mistakes during her attempt to overturn the democratic process in Congress on January 6, 2026:

1. Ignoring repeated lawful orders from Capitol Police.
2. Pushing to the front of the crowd.
3. Attempting to smash a window into the Speaker's Lobby.
4. Attempting to forcibly enter a barricaded door with weapons.

These actions demonstrate her disregard for authority and security protocols, which led to further chaos at the Capitol building. The...

[Latency: 15.60s]

============================================================
TOP_K = 3
============================================================
In the given context, there is no specific mention of any mistakes made by Elise Stefanik in Congress on January 23, 2026. Therefore, the correct answer is:

"There is insufficient information in the provided context to determine if Elise Stefanik made any mistakes in Congress on January 23, 2026."

[Latency: 2.99s]

============================================================
TOP_K = 5
============================================================
According to the context, Elise Stefanik made several mistakes in Congress on January 23, 2026:

1. She tried to overturn our democracy by ignoring repeated lawful orders from Capitol Police.
2. She pushed to the front of the crowd.
3. She helped to smash a window into the Speaker's Lobby.
4. She attempted to forcibly enter a barricaded door while being armed.

These actions suggest that she engaged in disruptive behavior during a highly sensitive political event at the Capitol building. Her att...

[Latency: 6.53s]

============================================================
TOP_K = 10
============================================================
Based on the given context, there is no specific mention of Elise Stefanik making any mistakes in Congress on January 23, 2026. Therefore, the information provided does not allow for an accurate answer to the question posed. The text focuses on various legislative actions and events without detailing any particular incident involving Elise Stefanik's performance in Congress. To provide a meaningful answer, additional context about her actions during that day would be necessary.

[Latency: 4.19s]

============================================================
TOP_K = 1
============================================================
The purpose of the Main Street Parity Act is to modify criteria for loans for plant acquisition, construction, conversion or expansion. This act may be cited as the 'Main Street Parity Act'. According to section 2, this modification involves changing Section 502(3)(C) of the relevant legislation.

The full text states "This Act may be cited as the ‘‘Main Street Parity Act’’. " Therefore, the main purpose of the Main Street Parity Act is to amend loan criteria related to plant acquisitions, cons...

[Latency: 6.26s]

============================================================
TOP_K = 3
============================================================
The purpose of the Main Street Parity Act is to modify the criteria for loans for plant acquisition, construction, conversion, or expansion, ensuring they align with the 504 standardization programs by establishing an equity requirement of 10%. This act aims to provide small businesses across the country with a fair and equitable opportunity to access capital while reflecting modern economic and technological realities. Additionally, it seeks to address concerns about traditional lending practic...

[Latency: 4.16s]

============================================================
TOP_K = 5
============================================================
The purpose of the Main Street Parity Act is to modify the criteria for loans for plant acquisition, construction, conversion, or expansion, ensuring that these loans align more closely with the standards set by the 504 standardization programs. Specifically, it introduces an equity requirement of 10% for these types of loans. This act aims to provide a fairer and more accessible financial opportunity for small businesses across the country, reflecting modern economic and technological realities...

[Latency: 7.52s]

============================================================
TOP_K = 10
============================================================
The purpose of the Main Street Parity Act is to modify the criteria for loans for plant acquisition, construction, conversion or expansion, ensuring these loans align with the 504 standardization programs by setting an equity requirement at 10%. This act aims to bring these loans in line with the existing economic and technological realities, offering a new opportunity for Americans across the country to access capital for their small businesses.

[Latency: 3.80s]

============================================================
TOP_K = 1
============================================================
The context mentions that there were efforts by the Biden administration to block States from partnering with pregnancy centers, but it also states that a bill was introduced that "pushes back" and protects a State's ability to partner with these centers. This suggests that some members of Congress may be supporting or opposing funding for pregnancy centers. However, without specific names mentioned, we cannot determine who exactly is speaking for or against funding. Therefore, based solely on t...

[Latency: 5.91s]

============================================================
TOP_K = 3
============================================================
According to the context, Ms. Dexter of Oregon spoke in favor of funding pregnancy centers during her speech addressing the House for 5 minutes. The context also mentions that the speaker opposed the proposal to divert money from families in need to anti-abortion crisis pregnancy centers. Based on these points, we can infer that Ms. Dexter supported funding for pregnancy centers while opposing the proposed redirection of funds. However, the exact details about who specifically spoke against fund...

[Latency: 4.47s]

============================================================
TOP_K = 5
============================================================
Based on the context provided, two individuals have spoken out against funding of pregnancy centers:

1. Representative Debbie Dingell (D-Michigan) spoke in favor of protecting States' ability to partner with trusted local organizations offering prenatal care, parenting classes, counseling, and other essential services.
2. Representative John Sarbanes (D-Maryland) spoke in opposition to the bill, calling it "irresponsible" and stating that crisis pregnancy centers are not medical facilities. He ...

[Latency: 7.24s]

============================================================
TOP_K = 10
============================================================
Representatives Michelle Fischerbach and Deb Haaland have spoken about funding for pregnancy centers. Representative Fischerbach supports the funding while Representative Haaland opposes it. The context mentions that Fischerbach's bill "Supporting Pregnant and Parenting Women and Families Act" passed, indicating her support. On the other hand, Haaland's speech criticizes the Biden administration's attempt to prevent funding for such centers, suggesting opposition. Both speakers highlight the imp...

[Latency: 4.48s]

Discussion:
For the question about what Mr. Flood had to say about Mayor David Black, the model was able to generate a "correct" answer that he was a "paragon of public service" since k=1. However, this was a short, basic answer that didn't go into what the specific impact he had on his town was, which only began to show up in k=5 onwards (nearly 17 1/2 years of service, emphasizing his leadership role in guiding Papillion into a thriving economic hub.) The response for k=5 and k=10 are relatively similar with the k=10 response having even more context about using partnerships with professionals and the business community along with his motivation for building a strong foundation for future generations. For the question about Elise Stefanik's mistake, it is unable to answer the question correctly at all k's and flip flops between hallucinating a mistake for "overturning the democracy" or stating that it is unable to answer the question based on the context, which can help us conclude that the document does not offer strong enough context to help answer this question (again, it is only one sentence where semantic matching is either on mistake or on her name which is hard so it isn't a problem with the RAG setup). Interestingly enough, for the question about the Main Street Parity Act, all the responses are pretty much identical across all k's. This finding is probably due to the most similar chunk being the most relevant one which the generation can be fully based on. Finally, on the question about pregnancy centers, we can also see the positive effects of having a larger k and gaining more context since the k=1 generation gives only an overview of the problem while the k=3 generation pinpoints one speaker's views and the k=5 and k=10 generations expanding to 2 contrasting speaker views. The generations with larger values of k greater than 1 are able to refer back to their context and give specific examples of what their views were. While these responses were accurate, I would argue that they aren't complete since they only provide 2 speakers instead of all members. However, this is due to an inability for providing the entire document as context (limited) and possibly the question wording. From these responses, we can conclude that the general pattern from these question/responses is that as we provide the model more context, we get more detailed responses. Interestingly enough, response latency is not a linear relationship when increasing k and instead jumps around seemingly randomly. Intuitively, we would expect that as k increases, latency would increase since we need to retrieve more neighbors and the context and therefore the input prompt increases so that the model takes longer to generate a response. Another line of thought could be that as we increase k, the model has more context so that it can give a succinct complete answer instead of rambling on and on since it is unsure how to answer the question and consequently shortening the latency for generating a response. However, we do not observe either of these behaviors and instead note that sometimes the latency for k=1 is higher than k=10 and vice versa with the k=3 and k=5 times also varying in relation to the others. With these examples there was no instance where too much context hurt the generations by confusing the model by providing irrelevant information, since the model generations were accurate for the most part. However, from this very limited collection of questions, I would argue that k=5 is the boundary where more context starts to have diminishing returns by citing the 1st and 4th question (since the 2nd and 3rd are bad examples). After k=5, the jump in improvement of generation decreases from previous jumps and the added context is minimal and does not add to the main idea of the generation. There is some interplay between k and chunk size since together they control how much context is provided in the prompt. There is a possible relationship where if we increase k and fetch more chunks, we could possibly decrease the size of chunks if we don't want to occupy the entire context window and maintain the same amount and vice versa. This setup could possibly affect our generations since by fetching more or less chunks, we are varying the scope of the corpus that we are providing to the model and could possibly introduce more valuable information to the model that is situated in different documents/chunks. Ideally, we would want to strike a balance where we adjust k and chunk size so that we con't add context to the point it hurts, but also want to provide enough relevant information so that the model is able to produce a good answer.

### Exercise 5: Handling Unanswerable Questions

I am using the Congressional Records corpus. I create 3 questions based on the different types of unanswerable questions and then generated responses before and after adjusting the prompt based on the experiment.

Unmodified Prompt:
Question (before prompt modification): What are the nutritional benefits of kale?

---

The context does not provide specific details about the nutritional benefits of kale. However, it mentions "micronutrients" including vitamins, which could be related to the health benefits of consuming kale. For precise information, one might need to refer to additional sources or studies dedicated to kale's nutritional value.

Relevant quote: "Micronutrients ......................................................."
"of which, Vitamin A ....................................."
"Vulnerable Children ......................................................."
"Blind Children ......................................................."
"HIV/AIDS ..........................................."

---

Question (before prompt modification): What did Congress discuss about healthcare on January 20, 2025?

---

Based on the given context, there is no specific mention of what Congress discussed about healthcare on January 20, 2025. The context focuses on discussions related to healthcare reform, particularly focusing on potential changes or issues with healthcare policies in place around January 8, 2026. There is no explicit reference to any discussion taking place on January 20, 2025. Therefore, the correct response is:

#### "There is insufficient information in the provided context to determine what Congress discussed about healthcare on January 20, 2025."

Question (before prompt modification): Why did Mr. Flood say only bad things about Mayor David Black?

---

Based solely on the information provided in the context, there is no indication why Mr. Flood said only negative things about Mayor David Black. The text does not provide any specific reasons or comments from Mr. Flood regarding Mayor Black's actions or accomplishments. Therefore, it cannot be determined from this context alone why Mr. Flood would have made such a statement. To answer this question accurately, we would need additional information beyond what is presented here. f

Modified Prompt:
Question (with modified prompt): What are the nutritional benefits of kale?

---

#### I cannot answer this from the available documents. Based on the given context, there is no mention of kale or its nutritional benefits. The text discusses grains, fruits, vegetables, and other types of food items, but does not provide specific information about kale's nutritional value. To answer this question accurately, additional sources beyond the provided context would be needed.

Question (with modified prompt): What did Congress discuss about healthcare on January 20, 2025?

---

#### I cannot answer this from the available documents.

Question (with modified prompt): Why did Mr. Flood say only bad things about Mayor David Black?

---

I cannot answer this from the available documents. The context does not provide any reason why Mr. Flood said only bad things about Mayor David Black. It only mentions that Mr. Flood recognized Mayor David Black as a paragon of public service and an outstanding public servant. There is no indication of negative remarks made by Mr. Flood regarding Mayor David Black. f

Discussion:
After testing a variety of unanswerable questions, a consistent behavior is that the model typically admits it doesn't know. This is because the prompt already says "If the context doesn't contain enough information to answer, say so". In the generated responses, I also didn't get any hallucinated wrong answers. Instead, the model made very broad remarks with what limited context it had, stretching information but then saying to refer to other sources. A good example is the question about the nutrition benefits of kale, where the generation with the unmodified prompt states that there aren't any specific benefits of kale mentioned but also stating micronutrients like vitamins was mentioned which could be related. The model even provides what evidence it sees in the form of 'Relevant quote: "Micronutrients ......................................................."
"of which, Vitamin A ....................................."
"Vulnerable Children ......................................................."
"Blind Children ......................................................."
"HIV/AIDS ..........................................."'. This is an instance that reinforces that in the context of these questions, the retrieved context didn't help (in the context of coming up with a good answer -> helped the model generation) nor hurt, but instead offered content for the model to build up a vague thought. In the question about healthcare, the model is even able to observe that "there is no specific mention of what Congress discussed about healthcare on January 20, 2025. The context focuses on discussions related to healthcare reform, particularly focusing on potential changes or issues with healthcare policies in place around January 8, 2026" and concludes that it has insufficient information. It admits it doesn't know, doesn't hallucinate, and doesn't use the retrieved context besides noticing the date. Finally, on the question about Mr. Flood saying bad things about Mayor David Black, the model says that there is no indication there were bad things said and cites that pulled context only reflects positive words. It admits it can't find examples of badmouthing, doesn't hallucinate, and uses the retrieved context to justify that only good things were said (helps). After modifying the prompt to tell the model to explicitly say "I cannot answer this from the available documents.", the model generates "I cannot answer this from the available documents" for ALL responses along with some justification in the first and third question. To further test unanswerable questions, I could come up with questions where I could guarantee there would be no similar topics to the text. For example, the question about healthcare from a different time would have fetched related context about healthcare from 2026 instead of 2025 which is unanswerable but shares similar content. However, I would argue that if the model is able to distinguish and clarify that it is unable to answer the question since the times don't match up, this is harder than answering an outright completely different problem and the model should be able to easily address that it is unable to answer the unanswerable question. I was surprised by how effective the tweak in the prompt was and did not know that instructing the model exactly what to say could short circuit the response. I would say that it helped since even though the model already recognized that it couldn't answer the questions, it would now explicitly say at the start that it couldn't.

### Exercise 6

#### Which phrasings retrieve the best chunks?

"When do I need to check the engine?" scored a 0.5496 and retrieved very specific chunks about engine diagnostics. "How often should I service the engine?" and "engine oil change lubrication schedule" scored 0.5085 and 0.5109 respectively, and were able to retrieve relevant information about oil change intervals, inspection schedules, and more. "Preventive maintenance requirements" did not perform as well with a score of 0.4125 and retrieved relatively less relevant chunks about dealer inventory and shop equipment.

#### Do keyword-style queries work better or worse than natural questions?

I found that the specific vocabulary mattered more than the style of querying. For example, for keyword-style, "Engine oil change lubrication schedule" scored very well, while "engine maintenance intervals" did not. On the other hand, for natural language question, "How often should I service the engine?" scored well, but doesn't seem to be because of the way the query was worded but rather the vocabulary used.

#### What does this tell you about potential query rewriting strategies?

When querying, it's best to use specific terminology found in the documents (like "oil change" and "lubrication") rather than abstract terms like "preventive maintenance requirements." In addition, some phrases like "When do I need to check the engine?" had a 0/5 overlap with the formal phrasing despite high scores. This shows that retrieval is very sensitive to wording.

### Exercise 7

#### Does higher overlap improve retrieval of complete information?

Higher overlap does not necessarily improve retrieval of complete information. overlap = 0 scored surprisingly well with a top score of 0.6185 and average top 5 score of 0.8070. However, overlap = 128 does better with a top score of 0.6044 and average top 5 score of 0.7024. In the middle though, overlap = 64 does worse than both of these with a top score of 0.7864 and average top 5 score of 0.8088. This shows that there is a sweet spot for the amount of overlap for retrieval of complete information.

#### What's the cost? (Index size, redundant information in context)

The cost of going from overlap = 0 to overlap = 256 resulted in over 2 times as many chunks (2320 to 5215) and over 2 times as much time to build the index (4.1s to 9.5s). In addition, when moving to Overlap 256, the information became redundant as ranks 2, 3, and 4 had almost identical text relating to cylinder head bolt holes (just with a slight offset). This wastes the LLM's context window with a lot of duplicate information.

#### Is there a point of diminishing returns?

Yes, as overlap = 128 seems to be the sweet spot with the top score of 0.6044 and best average top 5 score of 0.7024. However, increasing the overlap to 256 actually resulted in a lower top score and lower average top 5. In addition, it required almost double the chunk count without improving retrieval quality.

### Exercise 8

#### How does chunk size affect retrieval precision (relevant vs. irrelevant content)?

A smaller chunk size of 128 was able to retrieve more precise text but could not always capture the full context. A medium chunk size of 512 seemed to be the perfect balance between retrieving relevant information while still providing enough context to be useful. A large chunk size of 2048 seemed to hurt precision as it was able to retrieve relevant blocks of information, but a significant amount of irrelevant information with it.

#### How does it affect answer completeness?

Chunk size of 128 produced incomplete answers. While it would provide relevant information (or phrases), it wouldn't be of any substance. Chunk size of 512 produced the most complete answers. It was able to provide relevant information with multi-step procedures. Chunk size of 2048 produced longer answers but were often hallucinated or off topic. It would take in the irrelevant context and use it to create an answer that has nothing to do with the initial question.

#### Is there a sweet spot for your corpus?

A chunk size of 512 seemed to be the sweet spot for the Model T manual. It had the best balance of relevant retrieval and answer completeness. The summary scores also seem to reflect this as 512 had the best average top score.

#### Does optimal size depend on the type of question?

Yes the optimal size does depend on the type of question. For simple questions like "what oil should I use," a small chunk size like 128 is sufficient since the answer is short. However, for questions like asking for multi-step procedures, a chunk size of 512 was better as it was able to retrieve full blocks of procedures where a chunk size of 128 would have cut off. As stated before, a chunk size of 2048 gave too much context and would confuse the LLM with irrelevant information.

### Exercise 9

**Analysis of Similarity Scores (Top 10 Chunks)**

* **Score Distribution Patterns:**
    * **Clear Winners:** Queries like "Regulation (EU) 2024/1689" and "Model T Carburetor" showed a clear gap (~0.03–0.04) between the #1 and #2 results, indicating a highly specific semantic match.
    * **Clustered/Ambiguous:** Themes like "opening prayers" in the House resulted in very low, tightly clustered scores (around 0.48–0.50). This suggests the retriever found "vaguely religious" text across many documents but no "perfect" answer.
* **Threshold Recommendation:** Based on the logs, a **threshold of 0.60 to 0.65** is the sweet spot.
    * Above 0.65, results were almost always highly relevant.
    * Below 0.55, the pipeline started pulling "noise" (e.g., pulling Learjet engine data for a Model T question).
* **Effect of threshold on results:**  Using too low of a threshold resulted in unrelated sources being pulled in, which could negatively impact generation depending on how the question was prompted.

### Exercise 10 

**Overall Conclusion**

The results were pretty consistent across prompt templates in most circumstances, however there were a few instances where the various templates resulted in meaningful differences in output.

**Evaluation of System Prompts on Generation Quality**

- Accuracy: strict_grounding was the most accurate for preventing hallucinations. It forced the model to admit when information was missing rather than guessing.

- Usefulness: structured_output produced the most useful results. The explicit fact-listing phase acted made the final synthesis appear more systematic and grounded.

- The Trade-off: There is a clear tension between strict grounding and helpfulness. Strict grounding ensures accuracy but results in a "stiff" user experience (even resulting in one answer being totally imcomplete), whereas more permissive prompts feel more natural but introduces data not present in the corpora.

### Exercise 11

**Analysis of Model Performance Across k=3, k=5, and k=10**

* **Synthesis Capability:** The model successfully aggregated lists (e.g., safety warnings) from multiple files but failed at complex comparisons. When asked to compare procedures, it often concluded they were identical because the retrieved text lacked specific labels for comparison.
* **Impact of top_k:** At k=3, the model occasionally "forced" an answer from limited data. At k=10, the increased context allowed the model to self-correct and admit when specific information (like a monthly schedule) was actually missing from the manual.
* **Data Gaps:** Synthesis did not compensate for missing information. If a task wasn't in the top-k results, the model couldn't "reason" its existence. Retrieval noise at k=10 was generally ignored during the final synthesis step.

**Documented Findings**

* **Success:** The model is a reliable list aggregator for broad summaries but struggles with multi-hop logical tasks.
* **Missing Info:** The model is strictly bound by retrieval; it cannot synthesize facts that were not captured in the top-k chunks.
* **Conflicts:** No major contradictions occurred; the model often defaulted to a "not mentioned" or identical response when faced with ambiguity.
