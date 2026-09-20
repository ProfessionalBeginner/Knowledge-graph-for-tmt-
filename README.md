# Preface 
I am a highschool student, I don't have any real formal education in computer science; 
Full disclosure, the **implementation of the following idea is vibe coded**; with minimal manual testing; results show promise however a proper codebase audit is duly needed by someone that has far more skill than I do
This is an adaptation of an older project of mine that was since abandoned; the idea was to use a system similar to [graphrag](https://github.com/microsoft/graphrag) with SLMs and compare them to LLMs; however this was already done. 


# Core idea 
This is an addon to the following repo and has no real function without it: [jrz97619761/test-model-thing)](https://github.com/jrz97619761/test-model-thing)
Using a knowledge graph that can be traversed using algorithms such as (personalized) page rank [Page rank Wikipedia ](https://en.wikipedia.org/wiki/PageRank) 
This improves the context that the model has with its answers and information doesn't exclusively need to be encoded into weights; hence allowing training regimes to be focused more on core logic with bulk facts being far cheaper than adding millions-billions parameters to encode facts into weights
Rather tan acting as a simple graphrag; model generates query and then proceeds to get the response from the graph; it is designed deeper into the into the architecture. With the model's internal state being used for vector comparison with the graph, and traversed facts being reinjected into the model <br>
As an add-on it may be possible to use simple regex / property matching to create further relationships; and an advanced model may be able to 'create nodes to reason' (i understand this is a very vague statement however I am struggling to articulate it any better) <br> 
Having a second dimension of information on the relationships; such as relationship strengths / dependencies may also be significantly useful 

# Scope 
The scope of this repo is limited to the word being identified by the model and appropriately selecting the right nodes in the graph system 
A full response is limited by 2 factors, the first being the fact that the model has not been trained with any injected vectors and hence is not able to process them and the second being the small size + training set of the model leading to no real general understanding of the words 

# Scaling
A proper Knowledge base library should be used rather than a json file. 
Model needs to be trained with the graph not independently of it, and needs greater scale.

```
.venv/bin/python ask.py "what is a boat made of? "
==============================================================================
YOU:         'what is a boat made of? '#known bug, needs trailing space at the end of the string
locked on:   'boat' (at 's a boat', confidence 0.97) 
graph found: ['boat HasA sail', 'boat UsedFor sailing', 'boat IsA vehicle']
Graph off:   what is a boat made of? 'My sent at more at more at more at more at more '
Graph on:    what is a boat made of? 'benenenligeeeeGde. I say shat benenenligeyingeee'

```
As is visible, although the model is able to identify 'boat' and select the right node, the response is garbage. 

Following resource may be used for larger knowledge graphs: https://conceptnet.io
I have also developed a small tool that uses LLMs to generate knowledge graphs from PDFs / seed topics. However I do not believe that solution to be an appropriate as the purpose of using KBs is to improve accuracy and if they are based on LLM responses the accuracy may drop. 

# Contact 
**Discord:** `minihdmi_22349`
